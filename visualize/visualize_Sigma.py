
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


if __name__ == '__main__':
    sigma_inv = torch.load("dvi_data_cub_False_lossProxyNCA_distribution_loss_lr4e-1_quad_initialize_proxy1_tau0.0/ResNet_2048_Model/Epoch_40/proxy.pth")['sigma_inv']
    print(sigma_inv.shape)
    # ax = sns.heatmap(((1/F.softplus(sigma_inv))**2).detach().cpu().numpy()[:10, :100])
    ax = sns.heatmap(((1/sigma_inv)**2).detach().cpu().numpy()[:10, :100])
    plt.show()