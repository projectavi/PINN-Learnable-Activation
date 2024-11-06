import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS
from tqdm import tqdm

from util import *
from model.qres import QRes
from model.pinn import PINNs
from model.pinnsformer import PINNsformer

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def h(x):
    return np.exp(- (x - np.pi) ** 2 / (2 * (np.pi / 4) ** 2))


def u_ana(x, t):
    return h(x) * np.exp(5 * t) / (h(x) * np.exp(5 * t) + 1 - h(x))

if __name__ == "__main__":

    activations = ["nn.Sigmoid", "nn.Tanh", "WaveAct", "PeriodicWaveAct", "LearnableLeakyRelu", "Swish", "LAU"]

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    epochs = 50

    pinns = {}
    qres = {}
    pinnsformer = {}

    device = 'cuda:0'

    os.makedirs("./figs", exist_ok=True)

    # PINN - 1D Reaction

    res, b_left, b_right, b_upper, b_lower = get_data([0, 2 * np.pi], [0, 1], 101, 101)
    res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)

    res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
    b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
    b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
    b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
    b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

    x_res, t_res = res[:, 0:1], res[:, 1:2]
    x_left, t_left = b_left[:, 0:1], b_left[:, 1:2]
    x_right, t_right = b_right[:, 0:1], b_right[:, 1:2]
    x_upper, t_upper = b_upper[:, 0:1], b_upper[:, 1:2]
    x_lower, t_lower = b_lower[:, 0:1], b_lower[:, 1:2]

    for act in activations:
        model = PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4, activation=act).to(device)

        model.apply(init_weights)
        optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

        loss_track = []

        for i in tqdm(range(epochs)):
            def PINNclosure():
                pred_res = model(x_res, t_res)
                pred_left = model(x_left, t_left)
                pred_right = model(x_right, t_right)
                pred_upper = model(x_upper, t_upper)
                pred_lower = model(x_lower, t_lower)

                u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                          create_graph=True)[0]
                u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                          create_graph=True)[0]

                loss_res = torch.mean((u_t - 5 * pred_res * (1 - pred_res)) ** 2)
                loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
                loss_ic = torch.mean(
                    (pred_left[:, 0] - torch.exp(- (x_left[:, 0] - torch.pi) ** 2 / (2 * (torch.pi / 4) ** 2))) ** 2)

                loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

                loss = loss_res + loss_bc + loss_ic
                optim.zero_grad()
                loss.backward()
                return loss
            optim.step(PINNclosure)

        # Visualize PINNs
        res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
        x_test, t_test = res_test[:, 0:1], res_test[:, 1:2]

        with torch.no_grad():
            pred = model(x_test, t_test)[:, 0:1]
            pred = pred.cpu().detach().numpy()

        pred = pred.reshape(101, 101)

        res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)
        u = u_ana(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

        rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
        rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

        print('relative L1 error: {:4f}'.format(rl1))
        print('relative L2 error: {:4f}'.format(rl2))

        pinns[act] = {
            "loss": loss_track,
            "L1 error": rl1,
            "L2 error": rl2
        }

        plt.figure(figsize=(4, 3))
        plt.imshow(pred, extent=[0, np.pi * 2, 1, 0], aspect='auto')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Predicted u(x,t)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'./figs/1dreaction_pinns_pred_{act}.png')

        plt.figure(figsize=(4, 3))
        plt.imshow(u, extent=[0, np.pi * 2, 1, 0], aspect='auto')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Exact u(x,t)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./figs/1dreaction_exact.png')

        plt.figure(figsize=(4, 3))
        plt.imshow(np.abs(pred - u), extent=[0, np.pi * 2, 1, 0], aspect='auto')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Absolute Error')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'./figs/1dreaction_pinns_error_{act}.png')

    os.makedirs("./results", exist_ok=True)

    with open(f'./results/1dreaction_pinns.pkl', 'wb') as f:
        pickle.dump(pinns, f)

    # QRes - 1D Reaction

    for act in activations:
        model = QRes(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4, activation=act).to(device)

        model.apply(init_weights)
        optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

        for i in tqdm(range(epochs)):
            def Qresclosure():
                pred_res = model(x_res, t_res)
                pred_left = model(x_left, t_left)
                pred_right = model(x_right, t_right)
                pred_upper = model(x_upper, t_upper)
                pred_lower = model(x_lower, t_lower)

                u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                          create_graph=True)[0]
                u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                          create_graph=True)[0]

                loss_res = torch.mean((u_t - 5 * pred_res * (1 - pred_res)) ** 2)
                loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
                loss_ic = torch.mean(
                    (pred_left[:, 0] - torch.exp(- (x_left[:, 0] - torch.pi) ** 2 / (2 * (torch.pi / 4) ** 2))) ** 2)

                loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

                loss = loss_res + loss_bc + loss_ic
                optim.zero_grad()
                loss.backward()
                return loss

            optim.step(Qresclosure)

        # Visualize PINNs
        res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
        x_test, t_test = res_test[:, 0:1], res_test[:, 1:2]

        with torch.no_grad():
            pred = model(x_test, t_test)[:, 0:1]
            pred = pred.cpu().detach().numpy()

        pred = pred.reshape(101, 101)

        res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)
        u = u_ana(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

        rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
        rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

        print('relative L1 error: {:4f}'.format(rl1))
        print('relative L2 error: {:4f}'.format(rl2))

        qres[act] = {
            "loss": loss_track,
            "L1 error": rl1,
            "L2 error": rl2
        }

        plt.figure(figsize=(4, 3))
        plt.imshow(pred, extent=[0, np.pi * 2, 1, 0], aspect='auto')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Predicted u(x,t)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'./figs/1dreaction_qres_pred_{act}.png')

        plt.figure(figsize=(4, 3))
        plt.imshow(u, extent=[0, np.pi * 2, 1, 0], aspect='auto')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Exact u(x,t)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./figs/1dreaction_exact.png')

        plt.figure(figsize=(4, 3))
        plt.imshow(np.abs(pred - u), extent=[0, np.pi * 2, 1, 0], aspect='auto')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Absolute Error')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'./figs/1dreaction_qres_error_{act}.png')

    with open(f'./results/1dreaction_qres.pkl', 'wb') as f:
        pickle.dump(qres, f)

    # PINNsformer - 1D Reaction
    model = PINNsformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2).to(device)

    model.apply(init_weights)
    optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

    loss_track = []

    for i in tqdm(range(epochs)):
        def Pformerclosure():
            pred_res = model(x_res, t_res)
            pred_left = model(x_left, t_left)
            pred_right = model(x_right, t_right)
            pred_upper = model(x_upper, t_upper)
            pred_lower = model(x_lower, t_lower)

            u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                      create_graph=True)[0]
            u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                      create_graph=True)[0]

            loss_res = torch.mean((u_t - 5 * pred_res * (1 - pred_res)) ** 2)
            loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
            loss_ic = torch.mean(
                (pred_left[:, 0] - torch.exp(- (x_left[:, 0] - torch.pi) ** 2 / (2 * (torch.pi / 4) ** 2))) ** 2)

            loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

            loss = loss_res + loss_bc + loss_ic
            optim.zero_grad()
            loss.backward()
            return loss
        optim.step(Pformerclosure)

    # Visualize PINNsformer
    res_test = make_time_sequence(res_test, num_step=5, step=1e-4)
    res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
    x_test, t_test = res_test[:, :, 0:1], res_test[:, :, 1:2]

    with torch.no_grad():
        pred = model(x_test, t_test)[:, 0:1]
        pred = pred.cpu().detach().numpy()

    pred = pred.reshape(101, 101)


    res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)
    u = u_ana(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

    rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
    rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

    print('relative L1 error: {:4f}'.format(rl1))
    print('relative L2 error: {:4f}'.format(rl2))

    pinnsformer = {
        "loss": loss_track,
        "L1 error": rl1,
        "L2 error": rl2
    }

    plt.figure(figsize=(4, 3))
    plt.imshow(pred, extent=[0, np.pi * 2, 1, 0], aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Predicted u(x,t)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'./figs/1dreaction_pinnsformer_pred.png')

    plt.figure(figsize=(4, 3))
    plt.imshow(u, extent=[0, np.pi * 2, 1, 0], aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Exact u(x,t)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./figs/1dreaction_exact.png')

    plt.figure(figsize=(4, 3))
    plt.imshow(np.abs(pred - u), extent=[0, np.pi * 2, 1, 0], aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Absolute Error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'./figs/1dreaction_pinnsformer_error.png')

    with open(f'./results/1dreaction_pinnsformer.pkl', 'wb') as f:
        pickle.dump(pinnsformer, f)