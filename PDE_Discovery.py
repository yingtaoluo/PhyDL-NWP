# This code is for experimental purposes only. It aims to assess whether the explicit PDE terms used are consistent with patterns observed in the data.










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
parser.add_argument('--data', type=str, default='ningxia', help='choose a dataset, MIMIC3 or CMS')

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
    denominator = torch.sqrt(torch.sum(pred_anomaly**2) * torch.sum(label_anomaly**2))
    
    acc = numerator / denominator
    
    return acc


# class ModelArch(torch.nn.Module):
#     def __init__(self, layers):
#         super(ModelArch, self).__init__()
#         models = []
#         for idx in range(1, len(layers) - 1):
#             models.append(nn.Linear(layers[idx - 1], layers[idx], bias=True))
#             models.append(nn.Tanh())
#         models.append(nn.Linear(layers[len(layers) - 2], layers[len(layers) - 1], bias=True))
#         self.model = nn.Sequential(*models)

#     def forward(self, inputs):
#         return self.model(inputs)

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
    def __init__(self, X, data_train, X_f, X_val, data_val, lb, ub, lr, reg, model, records, lambda_w):
        super(PhysicsInformedNN, self).__init__()

        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        self.model = model.to(device)

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
        self.f_batch_size = args.batch_size*(x_f_ratio+1)

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
        self.parameter_list = list(self.model.parameters()) + [self.lambda_w]
        self.optimizer_lbfgs = torch.optim.LBFGS(self.parameter_list, max_iter=10000, history_size=50)
        self.NonZeroMask_w_tf = torch.ones((total_terms, 1)).to(device)
        self.threshold = records['adam_pre_valid_loss'][-1][0] if len(records['adam_pre_valid_loss']) > 0 else 1e8
        self.learning_rate = lr
        self.loss_f_ratio = 10  # 100
        self.L1_coef = reg
        self.criterion = nn.MSELoss(reduction='mean')
        # Specify the parameters you want to optimize
        self.optimizer_adam = torch.optim.Adam(self.parameter_list, lr=self.learning_rate)

        # Recording
        self.records = records

    def net_U(self, x, y, t):
        H = torch.cat([x, y, t], 1)
        H = 2.0 * (H - self.lb) / (self.ub - self.lb) - 1.0
        Y = self.model(H)  # (num_data, num_var)
        return Y

    def net_f(self, x, y, t):
        predict_data = self.net_U(x, y, t)
        uvw = [predict_data[:, i:i + 1] for i in range(num_targets)]

        derivatives, derivatives_description = [], []
        for i in range(num_targets):
            w_x = torch.autograd.grad(uvw[i], x, grad_outputs=torch.ones_like(uvw[i]), create_graph=True)[0]
            w_y = torch.autograd.grad(uvw[i], y, grad_outputs=torch.ones_like(uvw[i]), create_graph=True)[0]
            w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
            w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
            w_xy = torch.autograd.grad(w_x, y, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]

            if i == 0:
                derivatives.append(torch.ones_like(w_x))
                derivatives_description.append('')

            derivatives.extend([w_x, w_y, w_xx, w_yy, w_xy])
            derivatives_description.extend(['V{}_x'.format(i), 'V{}_y'.format(i), 'V{}_xx'.format(i),
                                            'V{}_yy'.format(i), 'V{}_xy'.format(i)])

        lib_fun, self.lib_descr = self.build_library(uvw, derivatives, derivatives_description, PolyOrder=2,
                                                     data_description=['V{}'.format(i) for i in range(num_targets)])

        w_t, f_w = [], []
        Phi = torch.cat(lib_fun, 1)
        for i in range(num_targets):
            time_deriv = torch.autograd.grad(uvw[i], t, grad_outputs=torch.ones_like(uvw[i]), create_graph=True)[0]
            w_t.append(time_deriv)
            f_w.append(time_deriv - torch.mm(Phi, (self.lambda_w[i] * self.NonZeroMask_w_tf)))

        w_t = torch.stack(w_t)
        f_w = torch.stack(f_w)

        return f_w, Phi, w_t

    def build_library(self, data, derivatives, derivatives_description, PolyOrder=2, data_description=None):
        ## polynomial terms
        P = PolyOrder
        lib_poly = [torch.ones_like(data[0])]
        lib_poly_descr = ['']  # it denotes '1'
        for i in range(len(data)):  # polynomial terms of univariable
            for j in range(1, P + 1):
                lib_poly.append(data[i] ** j)
                if j == 1:
                    lib_poly_descr.append(data_description[i])
                else:
                    lib_poly_descr.append(data_description[i] + "**" + str(j))

        # lib_poly.append(data[0] * data[1])
        # lib_poly_descr.append(data_description[0] + data_description[1])
        # lib_poly.append(data[0] * data[2])
        # lib_poly_descr.append(data_description[0] + data_description[2])
        # lib_poly.append(data[1] * data[2])
        # lib_poly_descr.append(data_description[1] + data_description[2])

        ## derivative terms
        lib_deri = derivatives
        lib_deri_descr = derivatives_description

        ## Multiplication of derivatives and polynomials (including the multiplication with '1')
        lib_poly_deri = []
        lib_poly_deri_descr = []
        for i in range(len(lib_poly)):
            for j in range(len(lib_deri)):
                lib_poly_deri.append(lib_poly[i] * lib_deri[j])
                lib_poly_deri_descr.append(lib_poly_descr[i] + lib_deri_descr[j])

        return lib_poly_deri, lib_poly_deri_descr

    def Adam_Training(self, num_epochs):
        train_record, valid_record = [], []

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

                predict_data = self.net_U(x_batch, y_batch, t_batch)
                f_w_pred, _, _ = self.net_f(x_f_batch, y_f_batch, t_f_batch)

                loss_unit, loss_f_w, loss_lambda_w = [], [], []
                rmse_unit, acc_unit = [], []
                for i in range(num_targets):
                    loss_unit.append(self.criterion(data_batch[i], predict_data[:, i:i + 1]))
                    loss_f_w.append(self.loss_f_ratio * torch.mean(torch.square(f_w_pred[i])))
                    loss_lambda_w.append(self.L1_coef * torch.norm(self.lambda_w[i], p=1))
                    rmse_unit.append(torch.sqrt(self.criterion(data_batch[i], predict_data[:, i:i + 1])))
                    acc_unit.append(ACC(data_batch[i], predict_data[:, i:i + 1]))

                losses = sum(loss_unit) + sum(loss_f_w) + sum(loss_lambda_w)
                accs = sum(acc_unit)
                rmses = sum(rmse_unit)  
                loss = losses  # torch.log(losses)

                loss.backward(retain_graph=True)
                self.optimizer_adam.step()
                self.optimizer_adam.zero_grad()

                if b % 20 == 0:
                    loss_unit_list = [to_npy(loss_unit[j]).item() for j in range(num_targets)]
                    loss_f_w_list = [to_npy(loss_f_w[j]).item() for j in range(num_targets)]
                    loss_lambda_w = [to_npy(loss_lambda_w[j]).item() for j in range(num_targets)]
                    train_record.append([to_npy(losses).item()] + loss_unit_list + loss_f_w_list + loss_lambda_w)

                    rmse_unit_list = [to_npy(rmse_unit[j]).item() for j in range(num_targets)]
                    self.records["train_rmse"].append([to_npy(rmses).item()] + rmse_unit_list)
                    acc_unit_list = [to_npy(acc_unit[j]).item() for j in range(num_targets)]
                    self.records["train_acc"].append([to_npy(accs).item()] + acc_unit_list)

                    print('The training loss: {}, data loss: {}, equation loss:{}, regularization loss:{}.'.
                          format(to_npy(losses).item(), loss_unit_list, loss_f_w_list, loss_lambda_w))
                    print('The training rmse: {}, data rmse: {}.\n'.format(to_npy(rmses).item(), rmse_unit_list))
                    print('The training acc: {}, data acc: {}.\n'.format(to_npy(accs).item(), acc_unit_list))

                    # validation  # does it need torch.no_grad?
                    self.model.eval()

                    predict_val = self.net_U(self.x_val, self.y_val, self.t_val)
                    # f_w, Phi, w_t = self.net_f(self.x_val, self.y_val, self.t_val)

                    loss_unit_val, rmse_unit_val, acc_unit_val = [], [], []
                    for i in range(num_targets):
                        loss_unit_val.append(self.criterion(self.data_val[i], predict_val[:, i:i + 1]))
                        rmse_unit_val.append(torch.sqrt(self.criterion(self.data_val[i], predict_val[:, i:i + 1])))
                        acc_unit_val.append(ACC(self.data_val[i], predict_val[:, i:i + 1]))

                    losses_val = sum(loss_unit_val)
                    rmse_val, acc_val = sum(rmse_unit_val), sum(acc_unit_val)
                    rmse_val_list = [to_npy(rmse_unit_val[j]).item() for j in range(num_targets)]
                    self.records["valid_rmse"].append([to_npy(rmse_val).item()] + rmse_val_list)
                    acc_val_list = [to_npy(acc_unit_val[j]).item() for j in range(num_targets)]
                    self.records["valid_acc"].append([to_npy(acc_val).item()] + acc_val_list)
                    # loss = torch.log(losses)

                    print('The validation rmse: {}, data rmse: {}.\n'.format(rmse_val, rmse_val_list))
                    print('The validation acc: {}, data acc: {}.\n'.format(acc_val, acc_val_list))

                    loss_unit_val_list = [to_npy(loss_unit_val[j]).item() for j in range(num_targets)]
                    valid_record.append([to_npy(losses_val).item()] + loss_unit_val_list)

                    print('The validation loss: {}, data loss: {}.'.
                          format(to_npy(losses_val).item(), loss_unit_val_list))
                    
                    lambda_w_pred = to_npy(self.lambda_w).reshape(-1, total_terms)

                    for i in range(num_targets):
                        disc_eq_temp = []
                        values_list = []  # List to store the values
                        for i_lib in range(len(self.lib_descr)):
                            if lambda_w_pred[i, i_lib] != 0:
                                weight_value = abs(lambda_w_pred[i, i_lib])
                                disc_eq_temp.append((weight_value, str(weight_value) + self.lib_descr[i_lib]))
                                values_list.append(weight_value)  # Append the value to the values list

                        # Sort the terms by the mean value in descending order
                        disc_eq_temp.sort(key=lambda x: x[0], reverse=True)
                        values_list.sort(reverse=True)  # Sort the values list in descending order

                        # Extract the sorted terms
                        disc_eq_temp = [term for _, term in disc_eq_temp][:20]

                        # Join the terms
                        disc_eq = '+'.join(disc_eq_temp)

                        # print('The discovered {}-th equation: w_t = '.format(i) + disc_eq + '\n')
                        # print(values_list[:20])

                    if self.threshold > losses_val:
                        self.threshold = losses_val
                        # save the model
                        torch.save({'state_dict': self.model.state_dict(),
                                    'lambda_w': self.lambda_w,
                                    'records': self.records},
                                   'checkpoints/{}.pth'.format(checkpoint_name))

        return train_record, valid_record

    def STRidge_Training(self, num_epochs):
        self.loss_f_ratio = 2.
        self.L1_coef = 0.
        self.optimizer_adam = torch.optim.Adam(self.parameter_list, lr=1e-4)

        for self.it in tqdm(range(6)):
            print('STRidge starts')
            self.callTrainSTRidge()

            # Adam optimizer
            print('Adam starts')
            self.records['adam_str_train_loss'], self.records['adam_str_valid_loss'] = self.Adam_Training(num_epochs)

        lambda_w = self.lambda_w.detach().cpu().numpy()
        NonZeroInd_w = np.nonzero(lambda_w)
        NonZeroMask_w = np.zeros_like(lambda_w)
        NonZeroMask_w[NonZeroInd_w] = 1
        self.NonZeroMask_w_tf = torch.tensor(NonZeroMask_w).to(device)

    def joint_training(self, adam_epo=5000, str_epo=1000, pt_adam_epo=5000):
        # Adam optimizer pre-training
        self.records['adam_pre_train_loss'], self.records['adam_pre_valid_loss'] = self.Adam_Training(adam_epo)

        self.STRidge_Training(str_epo)

        # Adam optimizer post-training
        self.records['adam_post_train_loss'], self.records['adam_post_valid_loss'] = self.Adam_Training(pt_adam_epo)

    def predict(self, X_star):
        x_star = torch.tensor(X_star[:, 0:1], requires_grad=True).to(device)
        y_star = torch.tensor(X_star[:, 1:2], requires_grad=True).to(device)
        t_star = torch.tensor(X_star[:, 2:3], requires_grad=True).to(device)

        u_star = self.u_pred(x_star, y_star, t_star)
        v_star = self.v_pred(x_star, y_star, t_star)
        w_star = self.w_pred(x_star, y_star, t_star)

        return u_star.cpu().data.numpy(), v_star.cpu().data.numpy(), w_star.cpu().data.numpy()

    def callTrainSTRidge(self):
        lam = 1e-5
        d_tol = 1
        maxit = 100
        STR_iters = 10

        l0_penalty = None

        normalize = 2
        split = 0.8
        print_best_tol = False

        # Process of lambda_u
        self.model.eval()
        Phi, w_t_pred = [], []
        for b in tqdm(range(len(self.x_f_batches))):
            x_f_batch = self.x_f_batches[b % len(self.x_f_batches)]
            y_f_batch = self.y_f_batches[b % len(self.y_f_batches)]
            t_f_batch = self.t_f_batches[b % len(self.t_f_batches)]
            _, phi, wt_pred = self.net_f(x_f_batch, y_f_batch, t_f_batch)
            Phi.append(to_npy(phi))
            w_t_pred.append(to_npy(wt_pred))

        Phi, w_t_pred = torch.stack(Phi), torch.stack(w_t_pred)
        pdb.set_trace()

        lambda_w2 = self.TrainSTRidge(Phi.detach().cpu().numpy(), w_t_pred.detach().cpu().numpy(), lam, d_tol,
                                      maxit, STR_iters, l0_penalty, normalize, split, print_best_tol)
        self.lambda_w = torch.nn.Parameter(lambda_w2.to(device), requires_grad=True)

    def TrainSTRidge(self, R, Ut, lam, d_tol, maxit, STR_iters=10, l0_penalty=None, normalize=2, split=0.8,
                     print_best_tol=False):
        # Split data into 80% training and 20% test, then search for the best tolderance.
        np.random.seed(0)  # for consistency
        n, _ = R.shape
        train = np.random.choice(n, int(n * split), replace=False)
        test = [i for i in np.arange(n) if i not in train]
        TrainR = R[train, :]
        TestR = R[test, :]
        TrainY = Ut[train, :]
        TestY = Ut[test, :]
        # D = TrainR.shape[1]

        # Set up the initial tolerance and l0 penalty
        d_tol = float(d_tol)
        tol = d_tol

        w_best = self.lambda_w.detach().cpu().numpy()

        # err_f = np.linalg.norm(TestY - TestR.dot(w_best), 2)
        err_f = np.mean((TestY - TestR.dot(w_best)) ** 2)

        if l0_penalty is None and self.it == 0:
            self.l0_penalty_0 = err_f
            l0_penalty = self.l0_penalty_0
        elif l0_penalty == None:
            l0_penalty = self.l0_penalty_0

        err_lambda = l0_penalty * torch.nonzero(self.lambda_w).size(0)
        err_best = err_f + err_lambda
        tol_best = 0

        self.records['str_eq_progress'].append([err_best, err_f, err_lambda, tol_best])

        # Now increase tolerance until test performance decreases
        for iter in range(maxit):

            # Get a set of coefficients and error
            w = self.STRidge(TrainR, TrainY, lam, STR_iters, tol, normalize=normalize)
            # err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
            err_f = np.mean((TestY - TestR.dot(w.numpy())) ** 2)
            err_lambda = l0_penalty * torch.nonzero(w).size(0)
            err = err_f + err_lambda

            # Has the accuracy improved?
            if err <= err_best:
                err_best = err
                w_best = w
                tol_best = tol
                tol = tol + d_tol
                self.records['str_eq_progress'].append([err_best, err_f, err_lambda, tol])

            else:
                tol = max([0, tol - 2 * d_tol])
                d_tol = 2 * d_tol / (maxit - iter)
                tol = tol + d_tol

        if print_best_tol:
            print("Optimal tolerance:", tol_best)

        return w_best

    def STRidge(self, X0, y, lam, maxit, tol, normalize=2, print_results=False):
        n, d = X0.shape
        X = torch.zeros((n, d))
        y = torch.tensor(y)
        # First normalize data
        if normalize != 0:
            Mreg = torch.zeros((d, 1))
            for i in range(0, d):
                Mreg[i] = 1.0 / torch.tensor(np.linalg.norm(X0[:, i], normalize))
                X[:, i] = Mreg[i] * X0[:, i]
        else:
            X = X0

        w_best = self.lambda_w.detach().cpu().numpy()

        # Inherit w from previous training
        w = w_best / Mreg  # assuming self.lambda_w is a PyTorch tensor

        num_relevant = d
        biginds = torch.where(torch.abs(w) > tol)[0]

        ridge_append_counter = 0

        lambda_history_STRidge = [Mreg * w]

        # Threshold and continue
        for j in range(maxit):

            # Figure out which items to cut out
            smallinds = torch.where(torch.abs(w) < tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]

            # If nothing changes then stop
            if num_relevant == len(new_biginds):
                break
            else:
                num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0:
                    if normalize != 0:
                        w = Mreg * w
                        lambda_history_STRidge.append(w)
                        ridge_append_counter += 1
                        return w
                    else:
                        lambda_history_STRidge.append(w)
                        ridge_append_counter += 1
                        return w
                else:
                    break
            biginds = new_biginds

            # Otherwise get a new guess
            w[smallinds] = 0

            if lam != 0:
                A = torch.mm(X[:, biginds].T, X[:, biginds]) + lam * torch.eye(len(biginds))
                B = torch.mm(X[:, biginds].T, y)
                w[biginds] = torch.linalg.lstsq(A, B).solution
                # w[biginds], _ = torch.lstsq(B, A)[:len(biginds)]
                # X = torch.lstsq(B, A).solution[:A.size(1)] conver to X = torch.linalg.lstsq(A, B).solution
                lambda_history_STRidge.append(Mreg * w)
                ridge_append_counter += 1
            else:
                A = X[:, biginds]
                B = y
                w[biginds] = torch.linalg.lstsq(A, B).solution
                lambda_history_STRidge.append(w)
                ridge_append_counter += 1

        # Now that we have the sparsity pattern, use standard least squares to get w
        if len(biginds) != 0:
            A = torch.mm(X[:, biginds].T, X[:, biginds]) + lam * torch.eye(len(biginds))
            B = torch.mm(X[:, biginds].T, y)
            w[biginds] = torch.linalg.lstsq(A, B).solution

        if normalize != 0:
            w = Mreg * w
            lambda_history_STRidge.append(w)
            ridge_append_counter += 1
            return w
        else:
            lambda_history_STRidge.append(w)
            ridge_append_counter += 1
            return w


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # load model arch
    num_targets = 8
    total_terms = 697  # 286
    layers = [3] + 8 * [args.hidden_size] + [num_targets]
    max_steps = 24*3
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
    N_valid_test = max_steps  # The last 24*7 timestamps for validation and test sets
    N_train = steps - N_valid_test  # The rest for training set

    # Training set
    X_train = X_star[X_star[:, 2] < N_train]
    data_train = data_star[:, X_star[:, 2] < N_train]

    # Validation and test sets
    X_valid_test = X_star[X_star[:, 2] >= N_train]
    data_valid_test = data_star[:, X_star[:, 2] >= N_train]

    # Randomly sample validation and test sets from the same set
    N_valid = int(X_valid_test.shape[0] * 0.5)
    idx_valid = np.random.choice(X_valid_test.shape[0], N_valid, replace=False)
    X_valid = X_valid_test[idx_valid]
    data_valid = data_valid_test[:, idx_valid]

    # Test set, which are the rest of validation/test set
    idx_test = np.setdiff1d(np.arange(X_valid_test.shape[0]), idx_valid, assume_unique=True)
    X_test = X_valid_test[idx_test]
    data_test = data_valid_test[:, idx_test]

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
    data_valid = data_valid + noise * np.std(data_valid) * np.random.randn(num_var, data_valid.shape[1],
                                                                           data_valid.shape[2])
    data_test = data_test + noise * np.std(data_test) * np.random.randn(num_var, data_test.shape[1],
                                                                        data_test.shape[2])

    # =============================================================================
    # model training
    # =============================================================================
    model = ModelArch(layers)
    load = False
    checkpoint_name = 'joint'
    if load:
        checkpoint = torch.load('checkpoints/{}.pth'.format(checkpoint_name))
        model.load_state_dict(checkpoint['state_dict'])
        lambda_w = checkpoint['lambda_w']
        records = checkpoint['records']
    else:
        records = {'adam_pre_train_loss': [], 'adam_pre_valid_loss': [],
                   'lbfgs_pre_train_loss': [], 'lbfgs_pre_valid_loss': [],
                   'adam_str_train_loss': [], 'adam_str_valid_loss': [],
                   'str_eq_progress': [], 'nonzero_mask': [],
                   'adam_post_train_loss': [], 'adam_post_valid_loss': [],
                   "train_rmse": [], "train_acc": [],
                   "valid_rmse": [], "valid_acc": []}
        lambda_w = torch.zeros([num_targets, total_terms, 1], dtype=torch.float32)

    Trainer = PhysicsInformedNN(X_train, data_train, X_f, X_valid, data_valid, lb, ub,
                                args.lr, args.reg, model, records, lambda_w)

    Trainer.joint_training()
