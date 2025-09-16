from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer import Trainer

import segmentation_models_pytorch as smp
from medpy import metric

from dataset import TrainDataset, ValDataset
from dataloader import TwoStreamBatchSampler
from util.path_utils import collect_training_paths, collect_validation_paths
from util.utils import DiceLoss, sigmoid_rampup
from evaluate.utils import recompone_overlap

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, pred, mask):
        # mask = mask.unsqueeze(dim=1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        smooth = 1
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        mask_flat = mask.view(size, -1)
        intersection = pred_flat * mask_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + mask_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return (wbce + dice_loss).mean()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, patch_outputs, output):
        bs = output.shape[0]
        cls = output.shape[1]
        psz = patch_outputs.shape[-1]
        cn = output.shape[-1] // psz

        patch_outputs = patch_outputs.reshape(bs, cn, cn, cls, psz, psz)
        output = output.reshape(bs, cls, cn, psz, cn, psz).permute(0, 2, 4, 1, 3, 5)

        p_output_soft = torch.sigmoid(patch_outputs)
        outputs_soft = torch.sigmoid(output)

        loss = torch.mean((p_output_soft - outputs_soft) ** 2, dim=(0, 3, 4, 5)).sum()

        return loss

class Net(smp.Unet):
    def __init__(self, in_c: int = 1, out_c: int = 1):
        super().__init__(in_channels=in_c, classes=out_c)
        self.seg_loss = BCEDiceLoss()
        self.aux_proj = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        feature_map = self.aux_proj(decoder_output)
        masks = self.segmentation_head(decoder_output)
        return feature_map, masks


class CariesSSLNet(LightningModule):
    def __init__(self, lr: float = 0.001, l_batch_size: int = 8, theta: float = 0.99, SSL: bool = True):
        super().__init__()
        self.semi_train = SSL
        self.learning_rate = lr
        self.p = theta
        self.max_epoch = 200
        self.l_batch_size = l_batch_size
        self.glob_step = 0

        # networks
        self.model = Net()

        # loss
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.dice_loss = DiceLoss()
        self.contrast_loss = ContrastiveLoss()
        self.consist_loss = ConsistencyLoss()

        #evaluate result
        self.eval_dict = dict({"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []})

    def cropImage(self, image):
        # input torch.Size([8, 1, 384, 384])
        patch_size = 128
        # torch.Size([288, 1, 64, 64])    
        image_patch = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).permute(
            0, 2, 3, 1, 4, 5).reshape(-1, 1, patch_size, patch_size) 
        
        return image_patch

    def reshapeFeatMap(self, feat_map_patch , batch_size):
        # proj_final torch.Size([288, 128, 1, 1]
        num = 3
        feat_map_patch = feat_map_patch.reshape(batch_size, num, num, 128 , 1, 1).permute(
            0, 3, 1, 4, 2, 5).reshape(8, 128, num, num)   # torch.Size([8, 128, 6, 6])
        return feat_map_patch


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.glob_step += 1
        image, label = batch
        batch_size = image.shape[0]
        feat_map, pred = self.model(image)
        image_patch = self.cropImage(image)
        feat_map_patch, pred_patch = self.model(image_patch)
        feat_map_patch_reshape = self.reshapeFeatMap(feat_map_patch, batch_size)

        bce_loss = self.bce_loss(pred[:self.l_batch_size], label[:self.l_batch_size])
        dice_loss = self.dice_loss(pred[:self.l_batch_size], label[:self.l_batch_size])
        seg_loss = 0.5 * (bce_loss + dice_loss)
        contrast_loss = 0
        consist_loss = 0
        if self.semi_train:
            if self.current_epoch < 100:
                contrast_loss = self.contrast_loss(feat_map, feat_map_patch_reshape)
            else:
                consist_loss = self.consist_loss(pred_patch, pred)
        weight = sigmoid_rampup(self.current_epoch, 200)
        self.log('train_contrast_loss', contrast_loss, on_step=False, on_epoch=True)
        self.log('train_consist_loss', consist_loss, on_step=False, on_epoch=True)
        self.log('train_seg_loss', seg_loss, on_step=False, on_epoch=True)
        return seg_loss + weight * (contrast_loss + consist_loss)

    def validation_step(self, batch, batch_idx):
        self.eval()
        imgs, gt = batch
        imgs = imgs.permute(1, 0, 2, 3)
        with torch.no_grad():
            _, outputs = self(imgs)
        pred = torch.sigmoid(outputs)
        pred_imgs = recompone_overlap(pred.cpu().numpy(), 768, 1536, 192, 192)
        pred_imgs = np.array(pred_imgs > 0.5).squeeze()
        gt = (gt > 0.5).cpu().numpy().squeeze()
        dice = metric.binary.dc(pred_imgs, gt)
        jc = metric.binary.jc(pred_imgs, gt)
        sen = metric.binary.sensitivity(pred_imgs, gt)
        pre = metric.binary.precision(pred_imgs, gt)
        spe = metric.binary.specificity(pred_imgs, gt)
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
        self.log('val_mean_iou', mean_iou)
        self.log('val_mean_dice', mean_dice)
        self.log('val_mean_spe', mean_spe)
        self.log('val_mean_pre', mean_pre)
        self.log('val_mean_sen', mean_sen)
        self.eval_dict = dict({"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []})


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        poly_learning_rate = lambda epoch: (1 - float(epoch) / self.max_epoch) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_learning_rate)
        return [optimizer], [scheduler]        
    
    # def on_train_epoch_end(self):
    #     sample_imgs = self.sup_data[0]
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("train image", grid)
    #     sample_imgs = self.pred.sigmoid()[0]
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("train label", grid)

    # def on_train_batch_end(self, outputs, batch, batch_idx, unused: int = 0):
    #     alpha = min(1 - 1 / (self.glob_step + 1), self.p)
    #     for para1, para2 in zip(self.model_stu.parameters(), self.model_tea.parameters()):
    #         para1 = alpha * para1 + (1 - alpha) * para2  

    # def on_validation_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)
    #     # log sampled images
    #     sample_imgs = self(z)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_process(model, train_loader, val_loader, max_epochs):
    tb_logger = pl_loggers.TensorBoardLogger(str(Path("Cariouslog") / "CLCC"))
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor='val_mean_dice',
                                        filename='CLCC10-{epoch:02d}-{val_mean_dice:.4f}',
                                        save_top_k=5,
                                        mode='max',
                                        save_weights_only=True)

    trainer = Trainer(max_epochs=max_epochs, logger=tb_logger,
                    gpus=1 if torch.cuda.is_available() else 0,
                    precision=16 if torch.cuda.is_available() else 32,
                    check_val_every_n_epoch=1, benchmark=True,
                    callbacks=[lr_monitor, checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_dataloaders=val_loader)


def main():
    SSL_flag = True
    learning_rate = 1e-3
    theta = 0.99
    labeled_ratio = {"0.1": 265, "0.2": 530, "0.5": 1325}
    labeled_rate = "0.1"
    if SSL_flag:
        batch_size, l_batch_size = 8, 4
    else:
        batch_size, l_batch_size = 4, 4

    project_root = Path.cwd()
    train_root = project_root / "data"
    train_image_list, train_label_list, ul_image_list = collect_training_paths(train_root)
    if SSL_flag and not ul_image_list:
        raise FileNotFoundError("No unlabeled images were found but SSL training was requested.")
    train_data = TrainDataset(train_image_list, train_label_list, ul_image_list if SSL_flag else None)

    panorama_root = project_root.parent / "caries_data" / "Max100Dice"
    panorama_img_path_list, panorama_gt_path_list = collect_validation_paths(panorama_root)
    val_data = ValDataset(panorama_img_path_list, panorama_gt_path_list)

    model = CariesSSLNet(learning_rate, l_batch_size, theta, SSL_flag)

    labeled_len = labeled_ratio[labeled_rate]
    if labeled_len > len(train_image_list):
        raise ValueError(f"Requested {labeled_len} labelled samples but only {len(train_image_list)} are available.")

    total_samples = len(train_image_list) + (len(ul_image_list) if SSL_flag else 0)
    labeled_idxs = list(range(labeled_len))
    unlabeled_idxs = list(range(len(train_image_list), total_samples)) if SSL_flag else []
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, l_batch_size)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0, pin_memory=pin_memory)

    max_epoch = 200
    train_process(model, train_loader, val_loader, max_epoch)


if __name__ == '__main__':
    seed_everything()
    main()

