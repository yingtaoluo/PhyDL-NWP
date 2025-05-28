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
import torch.utils.data
from pyDOE import lhs
from models import AFNONet, ConvLSTM
import argparse
from tqdm import tqdm
from main import ModelArch, get_derivative_value
import pdb
import os

lib_descr = []

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='convlstm', help='choose which model to train')
parser.add_argument('--output_mode', type=str, default='seq', help='choose to output a next step or many steps')
parser.add_argument('--load', type=bool, default=False, help='choose to load a pretrained model or train from scratch')
parser.add_argument('--seed', type=int, default=42, help='choose which seed to load')
parser.add_argument('--gpu', type=int, default=0, help='choose which gpu device to use')
parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
parser.add_argument('--data', type=str, default='ningxia', help='choose a dataset')

aux_args = parser.add_argument_group('auxiliary')
aux_args.add_argument('--batch_size', type=int, help='choose a hidden size')
aux_args.add_argument('--hidden_size', type=int, help='choose a hidden size')
aux_args.add_argument('--drop', type=float, help='choose a drop out rate')
aux_args.add_argument('--reg', type=float, help='choose a regularization coefficient')
parser.set_defaults(batch_size=64, hidden_size=60, drop=0, reg=1e-7)

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

GPU = args.gpu
device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
print(device)


def finite_diff_1d(u, dx, order):
    kernel = None

    if order == 1:
        # Central difference kernel: [-1, 0, 1] / (2*dx)
        kernel = torch.tensor([[-0.5, 0.0, 0.5]], dtype=u.dtype, device=u.device) / dx
    elif order == 2:
        # Second derivative: [1, -2, 1] / dx^2
        kernel = torch.tensor([[1.0, -2.0, 1.0]], dtype=u.dtype, device=u.device) / dx**2
    else:
        raise NotImplementedError(f"Order {order} not supported")

    kernel = kernel.view(1, 1, -1)  # (out_channels, in_channels, kernel_size)

    # reshape u: (B, 1, N)
    u = u.unsqueeze(1)
    out = F.conv1d(u, kernel, padding=1)
    return out.squeeze(1)


def FiniteDiff(u, dx, d):
    n = u.shape[0]
    ux = torch.zeros((n,), dtype=u.dtype, device=u.device)

    if d == 1:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - u[i - 1]) / (2 * dx)

        ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
        ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
        return ux

    if d == 2:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2

        ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
        ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
        return ux

    if d == 3:
        for i in range(2, n - 2):
            ux[i] = (u[i + 2] / 2 - u[i + 1] + u[i - 1] - u[i - 2] / 2) / dx ** 3

        ux[0] = (-2.5 * u[0] + 9 * u[1] - 12 * u[2] + 7 * u[3] - 1.5 * u[4]) / dx ** 3
        ux[1] = (-2.5 * u[1] + 9 * u[2] - 12 * u[3] + 7 * u[4] - 1.5 * u[5]) / dx ** 3
        ux[n - 1] = (2.5 * u[n - 1] - 9 * u[n - 2] + 12 * u[n - 3] - 7 * u[n - 4] + 1.5 * u[n - 5]) / dx ** 3
        ux[n - 2] = (2.5 * u[n - 2] - 9 * u[n - 3] + 12 * u[n - 4] - 7 * u[n - 5] + 1.5 * u[n - 6]) / dx ** 3
        return ux

    if d > 3:
        return FiniteDiff(FiniteDiff(u, dx, 3), dx, d - 3)


def to_npy(x):
    return x.cpu().data.numpy() if torch.cuda.is_available() else x.detach().numpy()


class Trainer:
    def __init__(self, data_train, data_val, data_test, lr, reg, model, records, lambda_w, Q):
        super(Trainer, self).__init__()

        x_train, y_train, coord_train = data_train
        x_val, y_val, coord_val = data_val

        self.model = model.to(device)
        self.lambda_w = lambda_w.to(device)
        self.Q = Q.to(device)

        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.coord_train = torch.tensor(coord_train, dtype=torch.float32)

        self.x_val = torch.tensor(x_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32)

        # Testing data
        self.data_test = torch.tensor(data_test, dtype=torch.float32)

        if args.output_mode == 'seq':
            self.multi_predict = nn.Linear(1, max_steps).to(device)

        # Optimizers
        self.threshold = records['valid_loss'][-1][0] if len(records['valid_loss']) > 0 else 1e8
        self.learning_rate = lr
        self.criterion = nn.MSELoss(reduction='mean')
        # Specify the parameters you want to optimize
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=reg)

        # Recording
        self.records = records

    def next_step_predict(self, inputs, labels):
        loss_all, prediction = [], []
        for b in range(len(inputs)):
            val_bs = inputs[b].shape[0]
            predict_val = self.model(inputs[b])
            if args.output_mode == 'seq':
                predict_val = self.multi_predict(predict_val.unsqueeze(-1))
            prediction.append(predict_val)
            loss_unit = []
            for i in range(num_targets):
                loss_unit.append(self.criterion(labels[b][:, :, :, i], predict_val[:, :, :, i]) * val_bs)
            loss_all.append(loss_unit)

        loss_list = (torch.sum(torch.tensor(loss_all), 0) / self.x_val.shape[0]).detach().numpy().tolist()
        loss = sum(loss_list)

        return prediction, loss, loss_list

    def multi_step_predict(self, data):
        xs = [item[:, :back_window] for item in data]
        ys = [item[:, back_window] for item in data]
        losses, losses_all = 0, []
        for j in range(max_steps):
            prediction, loss, loss_list = self.next_step_predict(xs, ys)
            if j == 0:
                print('The test next step total loss: {}, individual data loss: {}.'.
                      format(loss, loss_list))
            losses += loss / max_steps
            losses_all.append(loss_list)
            if j < max_steps - 1:
                xs = [torch.cat((item[:, 1:], prediction[i].unsqueeze(1)), dim=1) for i, item in enumerate(xs)]
                ys = [item[:, back_window + j + 1] for item in data]
        losses_list = (torch.sum(torch.tensor(losses_all), 0) / max_steps).detach().numpy().tolist()

        return losses, losses_list

    # The PDE with latent force (Eq.2)
    def physics_loss(self, u, coord):
        PDE_error = []

        # u: (B, H, W, C, T), coord: (B, H, W, T, 3)
        dx = dy = dt = 1
        u = u.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C)
        coord = coord.permute(0, 3, 1, 2, 4)  # (B, T, H, W, 3)

        B, T, H, W, C = u.shape

        derivatives, derivatives_description = [], []
        for i in range(C):
            f = u[:, :, :, :, i]  # (B, T, H, W)

            # f_x: derivative along width (W axis)
            f_x = finite_diff_1d(f.reshape(-1, W), dx, order=1).reshape(f.shape)
            f_xx = finite_diff_1d(f.reshape(-1, W), dx, order=2).reshape(f.shape)
            # f_y: derivative along height (H axis)
            f_reshape_H = f.permute(0, 1, 3, 2).reshape(-1, H)  # (BTW, H)
            f_y = finite_diff_1d(f_reshape_H, dy, order=1).reshape(B, T, W, H).permute(0, 1, 3, 2)
            f_yy = finite_diff_1d(f_reshape_H, dy, order=2).reshape(B, T, W, H).permute(0, 1, 3, 2)
            # f_T: derivative along time (T axis)
            f_reshape_T = f.permute(0, 2, 3, 1).reshape(-1, T)  # (BHW, T)
            f_t = finite_diff_1d(f_reshape_T, dt, order=1).reshape(B, H, W, T).permute(0, 3, 1, 2)
            f_tt = finite_diff_1d(f_reshape_T, dt, order=2).reshape(B, H, W, T).permute(0, 3, 1, 2)

            # f_x = torch.zeros_like(f)
            # f_y = torch.zeros_like(f)
            # f_t = torch.zeros_like(f)
            # f_xx = torch.zeros_like(f)
            # f_yy = torch.zeros_like(f)
            # f_tt = torch.zeros_like(f)

            # for b in range(B):
            #     for j in range(W):
            #         f_x[b, :, j] = FiniteDiff(f[b, :, j], dx, 1)
            #         f_xx[b, :, j] = FiniteDiff(f[b, :, j], dx, 2)
            #     for i_h in range(H):
            #         f_y[b, i_h, :] = FiniteDiff(f[b, i_h, :], dy, 1)
            #         f_yy[b, i_h, :] = FiniteDiff(f[b, i_h, :], dy, 2)
            #
            # for i_h in range(H):
            #     for j in range(W):
            #         f_t[:, i_h, j] = FiniteDiff(f[:, i_h, j], dt, 1)
            #         f_tt[:, i_h, j] = FiniteDiff(f[:, i_h, j], dt, 2)

            # f_i = f.reshape(-1)
            # f_x = f_x.reshape(-1)
            # f_y = f_y.reshape(-1)
            # f_t = f_t.reshape(-1)

            if i == 0:
                derivatives.append(torch.ones_like(f_x))
                derivatives_description.append('1')

            derivatives.extend([f, f_t, f_x, f_y, f_tt, f_xx, f_yy])
            derivatives_description.extend(['V{}'.format(i), 'V{}_t'.format(i), 'V{}_x'.format(i), 'V{}_y'.format(i),
                                            'V{}_tt'.format(i), 'V{}_xx'.format(i), 'V{}_yy'.format(i)])

        coord_flat = coord.reshape(B * H * W * T, 3)
        Q = self.Q(torch.tensor(coord_flat, dtype=torch.float32, device=device))
        Q = [Q[:, i:i + 1].reshape(B, T, H, W) for i in range(num_targets)]  # [(B, T, H, W)]

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

        return torch.stack(PDE_error).mean()

    def train(self, num_epochs=100, batch_size=args.batch_size):
        train_record, valid_record, test_record = [], [], []

        # Create list of shuffled indices
        indices = torch.randperm(self.x_train.size(0))
        indice_val = torch.randperm(self.x_val.size(0))
        indice_test = torch.randperm(self.data_test.size(0))

        # Divide data into batches
        x_batches = [self.x_train[indices[i * batch_size:(i + 1) * batch_size]].to(device) for i in
                     range(int(len(indices) / batch_size + 1))]
        y_batches = [self.y_train[indices[i * batch_size:(i + 1) * batch_size]].to(device) for i in
                     range(int(len(indices) / batch_size + 1))]
        coord_batches = [self.coord_train[indices[i * batch_size:(i + 1) * batch_size]].to(device) for i in
                         range(int(len(indices) / batch_size + 1))]
        x_val_batches = [self.x_val[indice_val[i * batch_size:(i + 1) * batch_size]].to(device) for i in
                         range(int(len(indice_val) / batch_size + 1))]
        y_val_batches = [self.y_val[indice_val[i * batch_size:(i + 1) * batch_size]].to(device) for i in
                         range(int(len(indice_val) / batch_size + 1))]
        data_test_batches = [self.data_test[indice_test[i * batch_size:(i + 1) * batch_size]].to(device) for i in
                             range(int(len(indice_test) / batch_size + 1))]

        for it in range(num_epochs):
            print('Epoch {}.'.format(it + 1))
            self.model.train()

            for b in tqdm(range(len(x_batches))):
                predict_data = self.model(x_batches[b])  # torch.Size([64, 10, 31, 31, 8])
                if args.output_mode == 'seq':
                    predict_data = self.multi_predict(predict_data.unsqueeze(-1))

                loss_unit = []
                for i in range(num_targets):
                    loss_unit.append(self.criterion(y_batches[b][:, :, :, i], predict_data[:, :, :, i]))
                losses = sum(loss_unit)
                # loss = losses  # torch.log(losses)

                physics_reg = self.physics_loss(predict_data, coord_batches[b])
                alpha = 0.01  # tuning weight
                loss = losses + alpha * physics_reg

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_unit_list = [to_npy(loss_unit[j]).item() for j in range(num_targets)]
                train_record.append([to_npy(losses).item()] + loss_unit_list)

                if b % 10 == 0:
                    print('The training loss: {}, data loss: {}.'.format(to_npy(losses).item(), loss_unit_list))

                    # validation
                    self.model.eval()
                    with torch.no_grad():
                        _, loss_val, loss_val_list = self.next_step_predict(x_val_batches, y_val_batches)
                        valid_record.append([loss] + loss_val_list)

                        print('The validation total loss: {}, individual data loss: {}.'.
                              format(loss_val, loss_val_list))

                        if self.threshold > loss_val:
                            self.threshold = loss_val
                            # save the model
                            torch.save({'state_dict': self.model.state_dict(),
                                        'records': self.records},
                                       'checkpoints/{}.pth'.format(checkpoint_name))
                        if args.output_mode == 'seq':
                            x_test_batches = [item[:, :back_window] for item in data_test_batches]
                            y_test_batches = [item[:, back_window:].permute((0, 2, 3, 4, 1)) for item in
                                              data_test_batches]
                            _, loss_test, loss_test_list = self.next_step_predict(x_test_batches, y_test_batches)
                        else:
                            loss_test, loss_test_list = self.multi_step_predict(data_test_batches)
                        test_record.append([loss_test] + loss_test_list)

                        print('The test total loss: {}, individual data loss: {}.'.
                              format(loss_test, loss_test_list))

        return train_record, valid_record, test_record


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # load model arch
    num_targets = 8
    back_window = 10  # hours (increase to improve performance)
    max_steps = 24  # hours (increase to longer-term forecasting)
    test_window = max_steps + back_window

    # load data
    xyt_data = np.load('./data/ningxia_real_train.npy')[:, :, 5:-5]
    steps, n, m, num_var = xyt_data.shape
    # original (8760, 31, 31, 8)

    xyt_normalized = np.zeros_like(xyt_data)

    for i in range(num_var):
        mean = np.mean(xyt_data[:, :, :, i])
        std = np.std(xyt_data[:, :, :, i])
        xyt_normalized[:, :, :, i] = (xyt_data[:, :, :, i] - mean) / std
        print(i, mean, std)

    xyt_test = xyt_normalized[int(steps * 0.95):]
    xyt_normalized = xyt_normalized[:int(steps * 0.95)]

    # ==== Construct spatiotemporal coordinates ====
    dt, dx, dy = 1, 1, 1
    steps, n, m, num_var = xyt_normalized.shape

    t_grid = np.arange(steps) * dt
    x_grid = np.arange(n) * dx
    y_grid = np.arange(m) * dy

    xv, yv, tv = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')  # (n, m, T)
    coords = np.stack([xv, yv, tv], axis=-1).transpose((2, 0, 1, 3))  # (T, n, m, 3)
    xyt_coords = coords  # shape: (T, n, m, 3)

    train_offset = np.sort(np.concatenate((np.arange(-back_window, 0, 1),)))
    if args.output_mode == 'seq':
        seq_offset = np.sort(np.concatenate((np.arange(0, max_steps, 1),)))

    # get samples
    x, y, coord_x, coord_y = [], [], [], []
    for t in range(back_window, len(xyt_normalized) - max_steps):
        x_t = xyt_normalized[t + train_offset, ...]  # (back_window, n, m, C)
        x_coord_t = xyt_coords[t + train_offset, ...]  # (back_window, n, m, 3)
        if args.output_mode == 'seq':
            y_t = xyt_normalized[t + seq_offset, ...].transpose((1, 2, 3, 0))  # (n, m, C, steps)
            y_coord_t = xyt_coords[t + seq_offset, ...].transpose((1, 2, 0, 3))  # (n, m, steps, 3)
        else:
            y_t = xyt_normalized[t, ...]  # (n, m, C)
            y_coord_t = xyt_coords[t, ...]  # (n, m, 3)
        x.append(x_t)
        y.append(y_t)
        coord_x.append(x_coord_t)
        coord_y.append(y_coord_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    coord_x = np.stack(coord_x, axis=0)
    coord_y = np.stack(coord_y, axis=0)
    # size (8750, 10, 31, 31, 8), (8750, 31, 31, 8)/(31, 31, 8, max_steps)

    num_samples = x.shape[0]
    # num_val = round(num_samples * 0.2)
    num_train = round(num_samples * 0.8)

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (x[num_train:], y[num_train:])
    # test
    # x_test, y_test = x[-num_test:], y[-num_test:]

    test_offset = np.sort(np.concatenate((np.arange(-test_window, 0, 1),)))

    # get samples
    test_seq = []
    for t in range(test_window, len(xyt_test)):
        seq_t = xyt_test[t + test_offset, ...]
        test_seq.append(seq_t)
    test_seq = np.stack(test_seq, axis=0)
    print("test sequence shape: ", test_seq.shape)  # (8750, 10+24*7, 31, 31, 8)

    layers = [3] + 8 * [100] + [num_targets]
    latent_force = ModelArch(layers)
    checkpoint_name = 'joint'
    checkpoint = torch.load('checkpoints/{}.pth'.format(checkpoint_name))
    latent_force.load_state_dict(checkpoint['state_dict_Q'])
    lambda_w = checkpoint['lambda_w']

    # =============================================================================
    # model training
    # =============================================================================
    if args.model == 'afno':
        model = AFNONet()
        checkpoint_name = 'afno'
    elif args.model == 'convlstm':
        model = ConvLSTM()
        checkpoint_name = 'convlstm'
    else:
        raise NotImplementedError()

    if args.load:
        checkpoint = torch.load('checkpoints/{}.pth'.format(checkpoint_name))
        model.load_state_dict(checkpoint['state_dict'])
        records = checkpoint['records']
    else:
        records = {'train_loss': [], 'valid_loss': []}

    trainer = Trainer((x_train, y_train, coord_y[:num_train]),
                      (x_val, y_val, coord_y[num_train:]),
                      test_seq, args.lr, args.reg, model, records,
                      lambda_w, latent_force)
    trainer.train()


