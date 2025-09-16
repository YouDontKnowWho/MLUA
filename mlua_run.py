from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from medpy import metric
from pytorch_lightning import LightningModule, loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import TrainDataset, ValDataset
from dataloader import TwoStreamBatchSampler
from evaluate.utils import recompone_overlap
from model.FPN import Net
from util.path_utils import collect_training_paths, collect_validation_paths
from util.utils import (
    DiceLoss,
    get_current_consistency_weight,
    mean_metric,
    sigmoid_mse_loss,
    sigmoid_rampup,
)


class CariesSSLNet(LightningModule):
    def __init__(self, lr: float = 0.001, l_batch_size: int = 8, theta: float = 0.99, ssl_enabled: bool = True) -> None:
        super().__init__()
        self.semi_train = ssl_enabled
        self.learning_rate = lr
        self.p = theta
        self.max_epoch = 200
        self.l_batch_size = l_batch_size
        self.glob_step = 0

        self.model_tea = Net()
        self.model_stu = Net()
        for para in self.model_tea.parameters():
            para.detach_()

        self.dice_loss = DiceLoss()
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.mse_loss = sigmoid_mse_loss

        self.eval_dict: Dict[str, List[float]] = {"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []}

    def forward(self, l_x: torch.Tensor) -> torch.Tensor:
        preds = self.model_stu(l_x)
        return preds[0]

    def training_step(self, batch, batch_idx):
        self.glob_step += 1
        imgs, gts = batch
        pred, pred_list = self.model_stu(imgs)
        consistency_loss = torch.tensor(0.0, device=imgs.device)

        if self.semi_train and imgs.shape[0] > self.l_batch_size:
            ul_data = imgs[self.l_batch_size:]
            noise = torch.clamp(torch.randn_like(ul_data) * 0.01, -0.1, 0.1)
            volume_batch_r = ul_data + noise
            with torch.no_grad():
                ul_pred = self.model_tea(volume_batch_r)[0]
            T = 8
            (_, c, h, w) = gts.shape
            stride = volume_batch_r.shape[0]
            preds = torch.zeros((T * 5 * stride, c, h, w), device=ul_pred.device, dtype=ul_pred.dtype)
            for i in range(T):
                ema_inputs = ul_data + torch.clamp(torch.randn_like(volume_batch_r) * 0.01, -0.1, 0.1)
                with torch.no_grad():
                    final_pred, pyramid_pred_list = self.model_tea(ema_inputs)
                    preds[20 * i : 20 * i + 4] = final_pred
                    pyramid_pred = torch.cat(pyramid_pred_list, dim=0)
                    preds[20 * i + 4 : 20 * i + 20] = pyramid_pred
            preds = preds.reshape(5 * T, stride, c, h, w)
            preds = torch.mean(preds, dim=0).sigmoid()
            uncertainty = -2.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            consistency_dist = self.mse_loss(pred[self.l_batch_size:], ul_pred)
            threshold = (0.75 + 0.25 * sigmoid_rampup(self.glob_step, 4480)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

        acc, iou, dice, spe, sen = mean_metric(pred[: self.l_batch_size], gts[: self.l_batch_size])
        self.log("train_mean_acc", acc, on_step=False, on_epoch=True)
        self.log("train_mean_iou", iou, on_step=False, on_epoch=True)
        self.log("train_mean_dice", dice, on_step=False, on_epoch=True)
        self.log("train_mean_spe", spe, on_step=False, on_epoch=True)
        self.log("train_mean_sen", sen, on_step=False, on_epoch=True)

        bce_loss = torch.tensor(0.0, device=imgs.device)
        dice_loss = torch.tensor(0.0, device=imgs.device)
        for pred_aux in pred_list:
            bce_loss += self.bce_loss(pred_aux[: self.l_batch_size], gts[: self.l_batch_size])
            dice_loss += self.dice_loss(pred_aux[: self.l_batch_size], gts[: self.l_batch_size])
        bce_loss += self.bce_loss(pred[: self.l_batch_size], gts[: self.l_batch_size])
        dice_loss += self.dice_loss(pred[: self.l_batch_size], gts[: self.l_batch_size])
        seg_loss = 0.5 * (bce_loss / 4 + dice_loss / 4)

        consistency_weight = get_current_consistency_weight(self.current_epoch, 200)

        self.log("train_consistency_loss", consistency_loss, on_step=False, on_epoch=True)
        self.log("train_bce_loss", bce_loss, on_step=False, on_epoch=True)
        self.log("train_dice_loss", dice_loss, on_step=False, on_epoch=True)
        self.log("train_seg_loss", seg_loss, on_step=False, on_epoch=True)
        return seg_loss + consistency_weight * consistency_loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 10 == 0 or self.current_epoch > 150:
            self.eval()
            imgs, gt = batch
            imgs = imgs.permute(1, 0, 2, 3)
            with torch.no_grad():
                outputs = self(imgs)
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

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % 10 == 0 or self.current_epoch > 150:
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
        else:
            self.log("val_mean_iou", 0.0)
            self.log("val_mean_dice", 0.0)
            self.log("val_mean_spe", 0.0)
            self.log("val_mean_pre", 0.0)
            self.log("val_mean_sen", 0.0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        poly_learning_rate = lambda epoch: (1 - float(epoch) / self.max_epoch) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_learning_rate)
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx, unused: int = 0):
        alpha = min(1 - 1 / (self.current_epoch + 1), self.p)
        for para1, para2 in zip(self.model_tea.parameters(), self.model_stu.parameters()):
            para1.data = alpha * para1.data + (1 - alpha) * para2.data


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_process(model: LightningModule, train_loader, val_loader, max_epochs: int, labeled_rate: str) -> None:
    model_names = {"0.1": "MLUA10", "0.2": "MLUA20", "0.5": "MLUA50"}
    model_name = model_names.get(labeled_rate, f"MLUA{labeled_rate.replace('.', '')}")

    log_dir = Path("Cariouslog") / "MULA"
    tb_logger = pl_loggers.TensorBoardLogger(str(log_dir))
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mean_dice",
        filename=model_name + "-{epoch:02d}-{val_mean_iou:.4f}-{val_mean_dice:.4f}-{val_mean_spe:.4f}-{val_mean_sen:.4f}-{val_mean_pre:.4f}",
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
    learning_rate = 1e-3
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
