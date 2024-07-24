from d2l import torch as d2l
from torch import nn
from torchvision import transforms

from load import valid_iter
from net import get_net
from train import epoch_update, get_labels


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        img = transforms.ToPILImage()(img)
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.tight_layout()
    return axes


def predict(net, test_iter, n=6):
    print(f"\n{'Epoch':>10}{'Loss':>10}{'Acc':>10}{'Diff':>10}")
    epoch_update(0, 1, net, test_iter, loss=nn.CrossEntropyLoss(), isTrain=False)

    for imgs, labels in test_iter:
        trues = get_labels(labels)
        preds = get_labels(net(imgs).argmax(dim=-1))
        break
    titles = [str(true) + "\n" + str(pred) for true, pred in zip(trues, preds)]
    show_images(imgs[0:n, :4].cpu(), 8, 8, titles=titles)


if __name__ == "__main__":
    net = get_net()
    predict(net, valid_iter, 64)
    d2l.plt.show()
