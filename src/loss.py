import torch.nn as nn
import torch.nn.functional as F


def get_loss(cfg):
    return MyLoss(cfg)


class MyLoss(nn.Module):
    def __init__(self, cfg):
        super(MyLoss, self).__init__()
        self.cfg = cfg
        self.loss = nn.KLDivLoss(reduction="batchmean") if not cfg.task.pretrain else nn.MSELoss()

    def forward(self, y_pred, y_true):
        return_dict = dict()

        if self.cfg.task.pretrain:
            loss = self.loss(y_pred, y_true)
            return_dict["loss"] = loss
            return return_dict

        y_pred = F.log_softmax(y_pred, dim=1)
        loss = self.loss(y_pred, y_true)
        return_dict["loss"] = loss
        return return_dict


def main():
    pass


if __name__ == '__main__':
    main()
