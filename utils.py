import numpy as np
import matplotlib.pyplot as plt

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def draw_training(epoch, loss_sup, loss_con, dice_train, loss_val, fig_name, lam):
    t = np.arange(1, epoch + 1, 1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(t, np.add(np.array(loss_con), lam * np.array(loss_sup)), color=color, label='total loss')
    ax1.plot(t, loss_sup, color='green', label='supervise loss')
    ax1.plot(t, loss_con, color='black', label='contrastive loss')
    ax1.plot(t, loss_val, color='brown', label='loss_val')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('dice', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, dice_train, color=color, label='dice_train')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"../models/{fig_name}/{fig_name}")
    plt.close()

def draw_training_joint_on_source(epoch, loss_sup, loss_con, dice_train, dice_val, con_val, fig_name, lam):
    t = np.arange(1, epoch + 1, 1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(t, np.add(np.array(loss_con), np.array(loss_sup)), color=color, label='train total loss')
    ax1.plot(t, loss_sup, color='green', label='train sup loss')
    ax1.plot(t, loss_con, color='black', label='train con loss')
    ax1.plot(t, con_val, color='yellow', label='val total loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('dice', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, dice_train, color=color, label='dice_train')
    ax2.plot(t, dice_val, color='brown', label='dice_val')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"../models/{fig_name}/{fig_name}")
    plt.close()

def draw_training_supervise(epoch, loss_sup, dice_train, dice_val, fig_name):
    t = np.arange(1, epoch + 1, 1)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(t, loss_sup, color='green', label='supervise loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('dice', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, dice_train, color=color, label='dice_train')
    ax2.plot(t, dice_val, color='brown', label='dice_val')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"../models/{fig_name}/{fig_name}")
    plt.close()

def draw_training_loss(epoch, loss):
    t = np.arange(1, epoch + 1, 1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(t, loss, color='green', label='loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("../models/train_curve_contrastive.png")
    plt.close()
