from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from medpy import metric
from pytorch_lightning import LightningModule, loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import TrainDataset, ValDataset
from dataloader import TwoStreamBatchSampler
from evaluate.utils import recompone_overlap
from model.smpFPN import FPNnet
from util.path_utils import collect_training_paths, collect_validation_paths
from util.utils import DiceLoss, get_current_consistency_weight, sigmoid_mse_loss


class CariesSSLNet(LightningModule):
    def __init__(self, lr: float = 0.001, l_batch_size: int = 8, theta: float = 0.99, ssl_enabled: bool = True):
        super().__init__()
        self.semi_train = ssl_enabled
        self.learning_rate = lr
        self.p = theta
        self.max_epoch = 200
        self.l_batch_size = l_batch_size
        self.glob_step = 0

        self.model = FPNnet(in_c=1, c=1)

        self.dice_loss = DiceLoss()
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.mse_loss = sigmoid_mse_loss
        self.kl_dist = nn.KLDivLoss(reduction="none")
        self.eval_dict: Dict[str, List[float]] = {"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.glob_step += 1
        volume_batch, label_batch = batch
        _, [outputs_aux3, outputs_aux2, outputs_aux1, outputs] = self.model(volume_batch)
        outputs_soft = torch.sigmoid(outputs)
        outputs_aux1_soft = torch.sigmoid(outputs_aux1)
        outputs_aux2_soft = torch.sigmoid(outputs_aux2)
        outputs_aux3_soft = torch.sigmoid(outputs_aux3)

        loss_ce = self.bce_loss(outputs[: self.l_batch_size], label_batch[: self.l_batch_size])
        loss_ce_aux1 = self.bce_loss(outputs_aux1[: self.l_batch_size], label_batch[: self.l_batch_size])
        loss_ce_aux2 = self.bce_loss(outputs_aux2[: self.l_batch_size], label_batch[: self.l_batch_size])
        loss_ce_aux3 = self.bce_loss(outputs_aux3[: self.l_batch_size], label_batch[: self.l_batch_size])

        loss_dice = self.dice_loss(outputs_soft[: self.l_batch_size], label_batch[: self.l_batch_size])
        loss_dice_aux1 = self.dice_loss(outputs_aux1_soft[: self.l_batch_size], label_batch[: self.l_batch_size])
        loss_dice_aux2 = self.dice_loss(outputs_aux2_soft[: self.l_batch_size], label_batch[: self.l_batch_size])
        loss_dice_aux3 = self.dice_loss(outputs_aux3_soft[: self.l_batch_size], label_batch[: self.l_batch_size])

        supervised_loss = (
            loss_ce
            + loss_ce_aux1
            + loss_ce_aux2
            + loss_ce_aux3
            + loss_dice
            + loss_dice_aux1
            + loss_dice_aux2
            + loss_dice_aux3
        ) / 8

        preds = (outputs_soft + outputs_aux1_soft + outputs_aux2_soft + outputs_aux3_soft) / 4

        variance_main = torch.sum(
            self.kl_dist(torch.log(outputs_soft[: self.l_batch_size]), preds[: self.l_batch_size]), dim=1, keepdim=True
        )
        exp_variance_main = torch.exp(-variance_main)

        variance_aux1 = torch.sum(
            self.kl_dist(torch.log(outputs_aux1_soft[: self.l_batch_size]), preds[: self.l_batch_size]), dim=1, keepdim=True
        )
        exp_variance_aux1 = torch.exp(-variance_aux1)

        variance_aux2 = torch.sum(
            self.kl_dist(torch.log(outputs_aux2_soft[: self.l_batch_size]), preds[: self.l_batch_size]), dim=1, keepdim=True
        )
        exp_variance_aux2 = torch.exp(-variance_aux2)

        variance_aux3 = torch.sum(
            self.kl_dist(torch.log(outputs_aux3_soft[: self.l_batch_size]), preds[: self.l_batch_size]), dim=1, keepdim=True
        )
        exp_variance_aux3 = torch.exp(-variance_aux3)

        consistency_dist_main = (preds[: self.l_batch_size] - outputs_soft[: self.l_batch_size]) ** 2
        consistency_loss_main = torch.mean(consistency_dist_main * exp_variance_main) / (
            torch.mean(exp_variance_main) + 1e-8
        ) + torch.mean(variance_main)

        consistency_dist_aux1 = (preds[: self.l_batch_size] - outputs_aux1_soft[: self.l_batch_size]) ** 2
        consistency_loss_aux1 = torch.mean(consistency_dist_aux1 * exp_variance_aux1) / (
            torch.mean(exp_variance_aux1) + 1e-8
        ) + torch.mean(variance_aux1)

        consistency_dist_aux2 = (preds[: self.l_batch_size] - outputs_aux2_soft[: self.l_batch_size]) ** 2
        consistency_loss_aux2 = torch.mean(consistency_dist_aux2 * exp_variance_aux2) / (
            torch.mean(exp_variance_aux2) + 1e-8
        ) + torch.mean(variance_aux2)

        consistency_dist_aux3 = (preds[: self.l_batch_size] - outputs_aux3_soft[: self.l_batch_size]) ** 2
        consistency_loss_aux3 = torch.mean(consistency_dist_aux3 * exp_variance_aux3) / (
            torch.mean(exp_variance_aux3) + 1e-8
        ) + torch.mean(variance_aux3)

        consistency_loss = (
            consistency_loss_main + consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3
        ) / 4
        consistency_weight = get_current_consistency_weight(self.current_epoch, 200)
        loss = supervised_loss + consistency_weight * consistency_loss

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        imgs, gt = batch
        imgs = imgs.permute(1, 0, 2, 3)
        with torch.no_grad():
            outputs = self(imgs)[1][-1]
        pred = torch.sigmoid(outputs)
        pred_imgs = recompone_overlap(pred.cpu().numpy(), 768, 1536, 192, 192)
        pred_imgs = np.array(pred_imgs > 0.5).squeeze()
        gt_np = (gt > 0.5).cpu().numpy().squeeze()
        dice = metric.binary.dc(pred_imgs, gt_np)
        jc = metric.binary.jc(pred_imgs, gt_np)
        sen = metric.binary.sensitivity(pred_imgs, gt_np)
        pre = metric.binary.precision(pred_imgs, gt_np)
        spe = metric.binary.specificity(pred_imgs, gt_np)
        self.eval_dict["iou"].append(jc)
        self.eval_dict["dice"].append(dice)
        self.eval_dict["pre"].append(pre)
        self.eval_dict["sen"].append(sen)
        self.eval_dict["spe"].append(spe)

    def on_validation_epoch_end(self):
        count = len(self.eval_dict["dice"])
        if count:
            mean_iou = float(np.mean(self.eval_dict["iou"]))
            mean_dice = float(np.mean(self.eval_dict["dice"]))
            mean_spe = float(np.mean(self.eval_dict["spe"]))
            mean_pre = float(np.mean(self.eval_dict["pre"]))
            mean_sen = float(np.mean(self.eval_dict["sen"]))
        else:
            mean_iou = mean_dice = mean_spe = mean_pre = mean_sen = 0.0
        self.log("val_mean_iou", mean_iou)
        self.log("val_mean_dice", mean_dice)
        self.log("val_mean_spe", mean_spe)
        self.log("val_mean_pre", mean_pre)
        self.log("val_mean_sen", mean_sen)
        self.eval_dict = {"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        poly_learning_rate = lambda epoch: (1 - float(epoch) / self.max_epoch) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_learning_rate)
        return [optimizer], [scheduler]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_process(model: LightningModule, train_loader, val_loader, max_epochs: int, labeled_rate: str) -> None:
    model_names = {"0.1": "URPC10", "0.2": "URPC20", "0.5": "URPC50"}
    model_name = model_names.get(labeled_rate, f"URPC{labeled_rate.replace('.', '')}")
    tb_logger = pl_loggers.TensorBoardLogger(str(Path("Cariouslog") / "URPC"))
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mean_dice",
        filename=model_name
        + "-{epoch:02d}-{val_mean_iou:.4f}-{val_mean_dice:.4f}-{val_mean_spe:.4f}-{val_mean_sen:.4f}-{val_mean_pre:.4f}",
        save_top_k=5,
        mode="max",
        save_weights_only=True,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=tb_logger,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,
        check_val_every_n_epoch=1,
        benchmark=True,
        callbacks=[lr_monitor, checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)


def main() -> None:
    ssl_flag = True
    learning_rate = 1e-4
    theta = 0.99
    labeled_ratio = {"0.1": 265, "0.2": 530, "0.5": 1325}
    labeled_rate = "0.1"

    if ssl_flag:
        batch_size, l_batch_size = 8, 4
    else:
        batch_size, l_batch_size = 4, 4

    project_root = Path.cwd()
    train_root = project_root / "data"
    train_images, train_labels, ul_images = collect_training_paths(train_root)
    if ssl_flag and not ul_images:
        raise FileNotFoundError("No unlabeled images were found but SSL training was requested.")
    train_data = TrainDataset(train_images, train_labels, ul_images if ssl_flag else None)

    panorama_root = project_root.parent / "caries_data" / "Max100Dice"
    panorama_imgs, panorama_gts = collect_validation_paths(panorama_root)
    val_data = ValDataset(panorama_imgs, panorama_gts)

    model = CariesSSLNet(learning_rate, l_batch_size, theta, ssl_flag)

    labeled_len = labeled_ratio[labeled_rate]
    if labeled_len > len(train_images):
        raise ValueError(f"Requested {labeled_len} labelled samples but only {len(train_images)} are available.")

    total_samples = len(train_images) + (len(ul_images) if ssl_flag else 0)
    labeled_indices = list(range(labeled_len))
    unlabeled_indices = list(range(len(train_images), total_samples)) if ssl_flag else []
    batch_sampler = TwoStreamBatchSampler(labeled_indices, unlabeled_indices, batch_size, l_batch_size)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_data,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        num_workers=0,
        pin_memory=pin_memory,
    )

    max_epoch = 200
    train_process(model, train_loader, val_loader, max_epoch, labeled_rate)


if __name__ == "__main__":
    seed_everything()
    main()
