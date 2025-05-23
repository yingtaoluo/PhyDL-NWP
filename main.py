# This contains the core code for understanding the logic and implementing PhyDL-NWP.

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import distance
import time
import torch.utils.data as data
from pyDOE import lhs
import argparse
from tqdm import tqdm
import pdb
import os

lib_descr = []

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ours', help='choose which model to train')
parser.add_argument('--seed', type=int, default=42, help='choose which seed to load')
parser.add_argument('--gpu', type=int, default=0, help='choose which gpu device to use')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--data', type=str, default='ningxia', help='choose from: ningxia, huadong, ningbo, weatherbench')

aux_args = parser.add_argument_group('auxiliary')
aux_args.add_argument('--batch_size', type=int, help='choose a batch size')
aux_args.add_argument('--hidden_size', type=int, help='choose a hidden size')
aux_args.add_argument('--drop', type=float, help='choose a drop out rate')
aux_args.add_argument('--reg', type=float, help='choose a regularization coefficient')
parser.set_defaults(batch_size=10000, hidden_size=100, drop=0, reg=1e-7)

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

GPU = args.gpu
device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
print(device)


def get_derivative_value(var_name, desc_list, values_list):
    try:
        idx = desc_list.index(var_name)
        return values_list[idx]
    except ValueError:
        return None  # or raise an error if preferred


def to_npy(x):
    return x.cpu().data.numpy() if torch.cuda.is_available() else x.detach().numpy()


def ACC(label, prediction):
    """
    Computes the Anomaly Correlation Coefficient between prediction and label tensors.

    Args:
    - prediction (torch.Tensor): The tensor containing predictions.
    - label (torch.Tensor): The tensor containing true labels.

    Returns:
    - acc (torch.Tensor): The Anomaly Correlation Coefficient.
    """

    # Ensure prediction and label have the same shape
    assert prediction.shape == label.shape, "Prediction and label tensors must have the same shape"

    # Calculate anomalies for prediction and label
    pred_anomaly = prediction - torch.mean(prediction)
    label_anomaly = label - torch.mean(label)

    # Compute ACC using the formula
    numerator = torch.sum(pred_anomaly * label_anomaly)
    denominator = torch.sqrt(torch.sum(pred_anomaly ** 2) * torch.sum(label_anomaly ** 2))

    acc = numerator / denominator

    return acc


class SinActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class ModelArch(torch.nn.Module):
    def __init__(self, layers):
        super(ModelArch, self).__init__()
        models = []
        for idx in range(1, len(layers) - 1):
            models.append(nn.Linear(layers[idx - 1], layers[idx], bias=True))
            models.append(SinActivation())  # use sin as activation function
        models.append(nn.Linear(layers[len(layers) - 2], layers[len(layers) - 1], bias=True))
        self.model = nn.Sequential(*models)

    def forward(self, inputs):
        return self.model(inputs)


class PhysicsInformedNN(torch.nn.Module):
    def __init__(self, X, data_train, X_f, X_val, data_val, lb, ub, lr, reg, model, Q, records, lambda_w):
        super(PhysicsInformedNN, self).__init__()

        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        self.model = model.to(device)
        self.Q = Q.to(device)

        # Parameters
        self.lambda_w = torch.nn.Parameter(lambda_w.to(device), requires_grad=True)

        # Training data
        self.x = torch.tensor(X[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y = torch.tensor(X[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.t = torch.tensor(X[:, 2:3], dtype=torch.float32, requires_grad=True).to(device)
        self.data = torch.tensor(data_train, dtype=torch.float32).to(device)

        # Collocation points
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 2:3], dtype=torch.float32, requires_grad=True).to(device)

        # Validation data
        self.x_val = torch.tensor(X_val[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_val = torch.tensor(X_val[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.t_val = torch.tensor(X_val[:, 2:3], dtype=torch.float32, requires_grad=True).to(device)
        self.data_val = torch.tensor(data_val, dtype=torch.float32).to(device)

        # Create list of shuffled indices
        indices = torch.randperm(self.x.size(0))
        f_indices = torch.randperm(self.x_f.size(0))
        self.f_batch_size = args.batch_size * (x_f_ratio + 1)

        # Divide data into batches
        self.x_batches = [self.x[indices[i * args.batch_size:(i + 1) * args.batch_size]] for i in
                          range(int(np.ceil(len(indices) / args.batch_size)))]
        self.y_batches = [self.y[indices[i * args.batch_size:(i + 1) * args.batch_size]] for i in
                          range(int(np.ceil(len(indices) / args.batch_size)))]
        self.t_batches = [self.t[indices[i * args.batch_size:(i + 1) * args.batch_size]] for i in
                          range(int(np.ceil(len(indices) / args.batch_size)))]
        self.data_batches = [self.data[:, indices[i * args.batch_size:(i + 1) * args.batch_size]] for i in
                             range(int(np.ceil(len(indices) / args.batch_size)))]

        self.x_f_batches = [self.x_f[f_indices[i * self.f_batch_size:(i + 1) * self.f_batch_size]] for i in
                            range(int(np.ceil(len(f_indices) / self.f_batch_size)))]
        self.y_f_batches = [self.y_f[f_indices[i * self.f_batch_size:(i + 1) * self.f_batch_size]] for i in
                            range(int(np.ceil(len(f_indices) / self.f_batch_size)))]
        self.t_f_batches = [self.t_f[f_indices[i * self.f_batch_size:(i + 1) * self.f_batch_size]] for i in
                            range(int(np.ceil(len(f_indices) / self.f_batch_size)))]

        # Optimizers
        self.parameter_list = list(self.model.parameters()) + list(self.Q.parameters()) + [self.lambda_w]
        self.threshold = records['val_rmse'][-1][0] if len(records['val_rmse']) > 0 else 1e8
        self.learning_rate = lr
        self.loss_f_ratio = 10  # 100
        self.L1_coef = reg
        self.criterion = nn.MSELoss(reduction='mean')
        # Specify the parameters you want to optimize
        self.optimizer_adam = torch.optim.Adam(self.parameter_list, lr=self.learning_rate)

        # Recording
        self.records = records

    # The "f(x, y, t)" that approximates the weather variables as a solution to the PDE net
    def net_f(self, x, y, t):
        H = torch.cat([x, y, t], 1)
        H = 2.0 * (H - self.lb) / (self.ub - self.lb) - 1.0
        Y = self.model(H)  # (num_data, num_var)
        return Y

    # The "Q(x, y, t)" as the latent force to complete the PDE
    def net_Q(self, x, y, t):
        H = torch.cat([x, y, t], 1)
        H = 2.0 * (H - self.lb) / (self.ub - self.lb) - 1.0
        Y = self.Q(H)  # (num_data, num_var)
        return Y

    # The PDE with latent force (Eq.2)
    def net_PDE(self, x, y, t):
        PDE_error = []
        predict_data = self.net_f(x, y, t)
        f = [predict_data[:, i:i + 1] for i in range(num_targets)]

        derivatives, derivatives_description = [], []
        for i in range(num_targets):
            w_t = torch.autograd.grad(f[i], t, grad_outputs=torch.ones_like(f[i]), create_graph=True)[0]
            w_x = torch.autograd.grad(f[i], x, grad_outputs=torch.ones_like(f[i]), create_graph=True)[0]
            w_y = torch.autograd.grad(f[i], y, grad_outputs=torch.ones_like(f[i]), create_graph=True)[0]
            w_tt = torch.autograd.grad(w_t, t, grad_outputs=torch.ones_like(w_t), create_graph=True)[0]
            w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
            w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
            # w_xy = torch.autograd.grad(w_x, y, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]

            if i == 0:
                derivatives.append(torch.ones_like(w_x))
                derivatives_description.append('1')

            derivatives.extend([f[i], w_t, w_x, w_y, w_tt, w_xx, w_yy])
            derivatives_description.extend(['V{}'.format(i), 'V{}_t'.format(i), 'V{}_x'.format(i), 'V{}_y'.format(i),
                                            'V{}_tt'.format(i), 'V{}_xx'.format(i), 'V{}_yy'.format(i)])

        Q = self.net_Q(x, y, t)
        Q = [Q[:, i:i + 1] for i in range(num_targets)]

        # 'V0_t + V0 * V0_x + V1 * V0_y - Q0 = 0'
        # 'V1_t + V0 * V1_x + V1 * V1_y - Q1 = 0'
        # 'V2_t + V2 * V2_x + V3 * V2_y - Q2 = 0'
        # 'V3_t + V2 * V3_x + V3 * V3_y - Q3 = 0'
        # 'V4_t + V0 * V4_x + V1 * V4_y - Q4 = 0'
        # 'V5_tt - v5_xx - v5_yy - Q5 = 0'
        # 'V6_tt - v6_xx - v6_yy - Q6 = 0'
        # 'V7_t + V0 * V7_x + V1 * V7_y - V7_xx - V7_yy - Q7 = 0'
        V0 = get_derivative_value('V0', derivatives_description, derivatives)
        V0_t = get_derivative_value('V0_t', derivatives_description, derivatives)
        V0_x = get_derivative_value('V0_x', derivatives_description, derivatives)
        V0_y = get_derivative_value('V0_y', derivatives_description, derivatives)
        V1 = get_derivative_value('V1', derivatives_description, derivatives)
        V1_t = get_derivative_value('V1_t', derivatives_description, derivatives)
        V1_x = get_derivative_value('V1_x', derivatives_description, derivatives)
        V1_y = get_derivative_value('V1_y', derivatives_description, derivatives)
        V2 = get_derivative_value('V2', derivatives_description, derivatives)
        V2_t = get_derivative_value('V2_t', derivatives_description, derivatives)
        V2_x = get_derivative_value('V2_x', derivatives_description, derivatives)
        V2_y = get_derivative_value('V2_y', derivatives_description, derivatives)
        V3 = get_derivative_value('V3', derivatives_description, derivatives)
        V3_t = get_derivative_value('V3_t', derivatives_description, derivatives)
        V3_x = get_derivative_value('V3_x', derivatives_description, derivatives)
        V3_y = get_derivative_value('V3_y', derivatives_description, derivatives)
        V4_t = get_derivative_value('V4_t', derivatives_description, derivatives)
        V4_x = get_derivative_value('V4_x', derivatives_description, derivatives)
        V4_y = get_derivative_value('V4_y', derivatives_description, derivatives)
        V5_tt = get_derivative_value('V5_tt', derivatives_description, derivatives)
        V5_xx = get_derivative_value('V5_xx', derivatives_description, derivatives)
        V5_yy = get_derivative_value('V5_yy', derivatives_description, derivatives)
        V6_tt = get_derivative_value('V6_tt', derivatives_description, derivatives)
        V6_xx = get_derivative_value('V6_xx', derivatives_description, derivatives)
        V6_yy = get_derivative_value('V6_yy', derivatives_description, derivatives)
        V7_t = get_derivative_value('V7_t', derivatives_description, derivatives)
        V7_x = get_derivative_value('V7_x', derivatives_description, derivatives)
        V7_y = get_derivative_value('V7_y', derivatives_description, derivatives)
        V7_xx = get_derivative_value('V7_xx', derivatives_description, derivatives)
        V7_yy = get_derivative_value('V7_yy', derivatives_description, derivatives)

        PDE_error.append(V0_t + self.lambda_w[0] * V0 * V0_x + self.lambda_w[1] * V1 * V0_y - Q[0])
        PDE_error.append(V1_t + self.lambda_w[2] * V0 * V1_x + self.lambda_w[3] * V1 * V1_y - Q[1])
        PDE_error.append(V2_t + self.lambda_w[4] * V2 * V2_x + self.lambda_w[5] * V3 * V2_y - Q[2])
        PDE_error.append(V3_t + self.lambda_w[6] * V2 * V3_x + self.lambda_w[7] * V3 * V3_y - Q[3])
        PDE_error.append(V4_t + self.lambda_w[8] * V0 * V4_x + self.lambda_w[9] * V1 * V4_y - Q[4])
        PDE_error.append(V5_tt - self.lambda_w[10] * V5_xx - self.lambda_w[11] * V5_yy - Q[5])
        PDE_error.append(V6_tt - self.lambda_w[12] * V6_xx - self.lambda_w[13] * V6_yy - Q[6])
        PDE_error.append(V7_t + self.lambda_w[14] * V0 * V7_x + self.lambda_w[15] * V1 * V7_y
                         - self.lambda_w[16] * V7_xx - self.lambda_w[17] * V7_yy - Q[7])

        return PDE_error

    def joint_training(self, num_epochs=10000):
        train_record, val_record = [], []

        for it in range(num_epochs):
            print('Epoch {}.'.format(it + 1))
            self.model.train()

            for b in tqdm(range(len(self.x_batches))):
                x_batch = self.x_batches[b]
                y_batch = self.y_batches[b]
                t_batch = self.t_batches[b]
                data_batch = self.data_batches[b]

                x_f_batch = self.x_f_batches[b % len(self.x_f_batches)]
                y_f_batch = self.y_f_batches[b % len(self.y_f_batches)]
                t_f_batch = self.t_f_batches[b % len(self.t_f_batches)]

                predict_data = self.net_f(x_batch, y_batch, t_batch)
                PDE_error = self.net_PDE(x_f_batch, y_f_batch, t_f_batch)

                loss_unit, loss_f_w = [], []
                rmse_unit, acc_unit = [], []
                for i in range(num_targets):
                    loss_unit.append(self.criterion(data_batch[i], predict_data[:, i:i + 1]))
                    loss_f_w.append(self.loss_f_ratio * torch.mean(torch.square(PDE_error[i])))  # loss for PDE
                    rmse_unit.append(torch.sqrt(self.criterion(data_batch[i], predict_data[:, i:i + 1])))
                    acc_unit.append(ACC(data_batch[i], predict_data[:, i:i + 1]))

                losses = sum(loss_unit) + sum(loss_f_w)
                accs = sum(acc_unit)
                rmses = sum(rmse_unit)
                loss = losses  # torch.log(losses)

                loss.backward(retain_graph=True)
                self.optimizer_adam.step()
                self.optimizer_adam.zero_grad()

                if b % 100 == 0:
                    loss_unit_list = [loss_unit[j].item() for j in range(num_targets)]
                    loss_f_w_list = [loss_f_w[j].item() for j in range(num_targets)]
                    train_record.append([losses.item()] + loss_unit_list + loss_f_w_list)

                    rmse_unit_list = [rmse_unit[j].item() for j in range(num_targets)]
                    self.records["train_rmse"].append([rmses.item()] + rmse_unit_list)
                    acc_unit_list = [acc_unit[j].item() for j in range(num_targets)]
                    self.records["train_acc"].append([accs.item()] + acc_unit_list)

                    print('The training loss: {}, where data loss is {}, and equation loss is {}.'.
                          format(losses.item(), loss_unit_list, loss_f_w_list))
                    print('The training rmse: {}, each variable: {}.\n'.format(rmses.item(), rmse_unit_list))
                    print('The training acc: {}, each variable: {}.\n'.format(accs.item(), acc_unit_list))

                    # validation  # does it need torch.no_grad?
                    self.model.eval()

                    predict_val = self.net_f(self.x_val, self.y_val, self.t_val)

                    loss_unit_val, rmse_unit_val, acc_unit_val = [], [], []
                    for i in range(num_targets):
                        loss_unit_val.append(self.criterion(self.data_val[i], predict_val[:, i:i + 1]))
                        rmse_unit_val.append(torch.sqrt(self.criterion(self.data_val[i], predict_val[:, i:i + 1])))
                        acc_unit_val.append(ACC(self.data_val[i], predict_val[:, i:i + 1]))

                    losses_val = sum(loss_unit_val)
                    rmse_val, acc_val = sum(rmse_unit_val), sum(acc_unit_val)
                    rmse_val_list = [rmse_unit_val[j].item() for j in range(num_targets)]
                    self.records["val_rmse"].append([rmse_val.item()] + rmse_val_list)
                    acc_val_list = [acc_unit_val[j].item() for j in range(num_targets)]
                    self.records["val_acc"].append([acc_val.item()] + acc_val_list)
                    # loss = torch.log(losses)

                    print('The validation rmse: {}, data rmse: {}.\n'.format(rmse_val, rmse_val_list))
                    print('The validation acc: {}, data acc: {}.\n'.format(acc_val, acc_val_list))

                    loss_unit_val_list = [loss_unit_val[j].item() for j in range(num_targets)]
                    val_record.append([losses_val.item()] + loss_unit_val_list)

                    print('The validation loss: {}, data loss: {}.'.
                          format(losses_val.item(), loss_unit_val_list))

                    if self.threshold > losses_val:
                        self.threshold = losses_val
                        # save the model
                        torch.save({'state_dict': self.model.state_dict(),
                                    'lambda_w': self.lambda_w,
                                    'records': self.records},
                                   'checkpoints/{}.pth'.format(checkpoint_name))

        return train_record, val_record


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # load model arch
    num_targets = 8
    layers = [3] + 8 * [args.hidden_size] + [num_targets]
    max_steps = 24 * 3
    x_f_ratio = 1

    # load data (some baselines can only process image-like data, so we set h=w)
    xyt_data = np.load('./data/ningxia_real_train.npy').transpose((3, 1, 2, 0))[:num_targets, :, 5:-5, :]
    num_var, n, m, steps = xyt_data.shape
    # original (8760, 31, 41, 8) -> (8, 31, 41, 8760)

    xyt_normalized = np.zeros_like(xyt_data)

    for i in range(num_var):
        mean = np.mean(xyt_data[i])
        std = np.std(xyt_data[i])
        print(i, mean, std)
        xyt_normalized[i] = (xyt_data[i] - mean) / std

    dt = 1
    dx = 1
    dy = 1

    # Preprocess data #1(First dimensiton is space and the second dimension is time.)
    st_data = xyt_normalized.reshape((num_var, n * m, steps))

    t_data = np.arange(steps).reshape((1, -1)) * dt
    t_data = np.tile(t_data, (m * n, 1))

    # This part reset the coordinates
    x_data = np.arange(n).reshape((-1, 1)) * dx
    x_data = np.tile(x_data, (1, m))
    x_data = np.reshape(x_data, (-1, 1))
    x_data = np.tile(x_data, (1, steps))

    y_data = np.arange(m).reshape((1, -1)) * dy
    y_data = np.tile(y_data, (n, 1))
    y_data = np.reshape(y_data, (-1, 1))
    y_data = np.tile(y_data, (1, steps))

    # Preprocess data #2(compatible with NN format)
    t_star = np.reshape(t_data, (-1, 1))
    x_star = np.reshape(x_data, (-1, 1))
    y_star = np.reshape(y_data, (-1, 1))
    data_star = np.reshape(st_data, (num_var, -1, 1))

    X_star = np.hstack((x_star, y_star, t_star))

    # Divide the dataset into training and validation/test sets
    N_val_test = max_steps  # The last 24*7 timestamps for validation and test sets
    N_train = steps - N_val_test  # The rest for training set

    # Training set
    X_train = X_star[X_star[:, 2] < N_train]
    data_train = data_star[:, X_star[:, 2] < N_train]

    # Validation and test sets
    X_val_test = X_star[X_star[:, 2] >= N_train]
    data_val_test = data_star[:, X_star[:, 2] >= N_train]

    # Randomly sample validation and test sets from the same set
    N_val = int(X_val_test.shape[0] * 0.5)
    idx_val = np.random.choice(X_val_test.shape[0], N_val, replace=False)
    X_val = X_val_test[idx_val]
    data_val = data_val_test[:, idx_val]

    # Test set, which are the rest of validation/test set
    idx_test = np.setdiff1d(np.arange(X_val_test.shape[0]), idx_val, assume_unique=True)
    X_test = X_val_test[idx_test]
    data_test = data_val_test[:, idx_test]

    # Bounds
    lb = np.min(X_star, 0)
    ub = np.max(X_star, 0)

    # Collocation points
    N_f = X_train.shape[0] * x_f_ratio
    X_f = lb + (ub - lb) * lhs(3, N_f)
    X_f = np.vstack((X_f, X_train))

    # add noise
    noise = 0.0
    data_train = data_train + noise * np.std(data_train) * np.random.randn(num_var, data_train.shape[1],
                                                                           data_train.shape[2])
    data_val = data_val + noise * np.std(data_val) * np.random.randn(num_var, data_val.shape[1],
                                                                           data_val.shape[2])
    data_test = data_test + noise * np.std(data_test) * np.random.randn(num_var, data_test.shape[1],
                                                                        data_test.shape[2])

    # =============================================================================
    # model training
    # =============================================================================
    model = ModelArch(layers)
    latent_force = ModelArch(layers)
    load = False
    checkpoint_name = 'joint'
    if load:
        checkpoint = torch.load('checkpoints/{}.pth'.format(checkpoint_name))
        model.load_state_dict(checkpoint['state_dict'])
        lambda_w = checkpoint['lambda_w']
        records = checkpoint['records']
    else:
        records = {"train_rmse": [], "val_rmse": [],
                   "train_acc": [], "val_acc": []}
        lambda_w = torch.zeros([18, 1], dtype=torch.float32)

    Trainer = PhysicsInformedNN(X_train, data_train, X_f, X_val, data_val, lb, ub,
                                args.lr, args.reg, model, latent_force, records, lambda_w)

    Trainer.joint_training()
