from dataclasses import asdict

import torch
from torch import nn
from tqdm import tqdm

import wandb
from load import test_dataset, test_iter, train_dataset, train_iter
from lr import LRManager
from net import get_net
from setting import num2Angle, setting
from utils import Timer

wandb.require("core")


def get_labels(y):
    return [num2Angle[int(num)] for num in y]


def format_str(tensor):
    return f"{tensor.item():>6.4f}"


def epoch_update(epoch, num_epochs, net, iter, loss, isTrain=True, lr=None):
    if isTrain:
        print(f"\n{'Epoch':>10}{'Loss':>10}{'Acc':>10}{'Diff':>10}")
    all_loss, all_acc, all_diffs = (
        torch.empty(0, device=setting.device),
        torch.empty(0, device=setting.device),
        torch.empty(0, device=setting.device),
    )
    if isTrain:
        net.train()
    else:
        net.eval()

    with tqdm(total=len(iter), ncols=100) as pbar:
        pbar.set_description(
            f"{str(epoch+1)+'/'+str(num_epochs):>10}{'0.0000':>10}{'0.0000':>10}{'0.0000':>10}"
        )
        for imgs, labels in iter:
            if isTrain:
                with lr.optim_step():
                    output = net(imgs)
                    ls = loss(output, labels)
                    ls.backward()
            else:
                with torch.no_grad():
                    output = net(imgs)
                    ls = loss(output, labels)

            preds = output.argmax(dim=-1)
            acc = (torch.abs(labels.float() - preds.float()) <= 2).float().mean()
            diff = (torch.abs(labels.float() - preds.float())).float().mean()
            all_diffs = torch.cat((all_diffs, diff.unsqueeze(0)))
            all_loss = torch.cat((all_loss, ls.unsqueeze(0)))
            all_acc = torch.cat((all_acc, acc.unsqueeze(0)))
            pbar.set_description(
                f"{str(epoch+1)+'/'+str(num_epochs):>10}{format_str(ls):>10}{format_str(acc):>10}{format_str(diff):>10}"
            )
            pbar.update()
    print(
        f"{'train'if isTrain else 'test':>10}{format_str(all_loss.mean()):>10}{format_str(all_acc.mean()):>10}{format_str(all_diffs.mean()):>10}"
    )
    return all_loss.mean(), all_acc.mean(), all_diffs.mean()


def train():
    net = get_net()
    if setting.is_log:
        wandb.config["net"] = type(net).__name__

    optimizer = torch.optim.SGD(
        net.parameters(), lr=setting.lr, momentum=setting.momentum
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=setting.lr,
        pct_start=setting.pct_start,
        epochs=setting.num_epochs,
        steps_per_epoch=len(train_iter),
    )
    lr = LRManager(setting.lr, scheduler, optimizer)
    loss = nn.CrossEntropyLoss()
    best_acc, best_diff = 0, 100
    global_step = 0

    print(f"训练集包含{len(train_dataset)}张图片，测试集包含{len(test_dataset)}张图片")
    print(f"训练设备{setting.device}")

    for epoch in range(setting.num_epochs):
        train_avgeloss, train_avgeacc, train_avgdiff = epoch_update(
            epoch, setting.num_epochs, net, train_iter, loss, isTrain=True, lr=lr
        )
        valid_avgeloss, valid_avgeacc, valid_avgdiff = epoch_update(
            epoch, setting.num_epochs, net, test_iter, loss, isTrain=False
        )

        if setting.is_log:
            wandb.log(
                {
                    "train_loss": train_avgeloss,
                    "train_acc": train_avgeacc,
                    "train_diff": train_avgdiff,
                    "valid_loss": valid_avgeloss,
                    "valid_acc": valid_avgeacc,
                    "valid_diff": valid_avgdiff,
                }
            )

        torch.save(net.state_dict(), setting.model_path / "last.pt")
        if valid_avgdiff < best_diff:
            best_diff = valid_avgdiff
            torch.save(net.state_dict(), setting.model_path / "best_diff.pt")

        if valid_avgeacc > best_acc:
            best_acc = valid_avgeacc
            global_step = 0
            torch.save(net.state_dict(), setting.model_path / "best_acc.pt")
        else:
            global_step += 1

        if global_step > 100:
            break


if __name__ == "__main__":
    if setting.is_log:
        wandb.init(project="RotNetW", config=asdict(setting))

    timer = Timer()
    timer.start()
    train()
    timer_s = timer.stop()
    print(f"{timer_s // 60}m{(timer_s % 60):.2f}s")
