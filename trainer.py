import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from metann.proto import ProtoModule

from network.model import MetaFD
from utils.data_helper import dg_dataloader


class MetaFDTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_number}" if torch.cuda.is_available() else "cpu")
        self._setup_paths()
        self.model = None
        self.optimizers = {}
        self.best_model_wts = None
        self.best_loss = float('inf')

        self.history = {
            'train_loss': [], 'val_loss': [],
            'epoch_times': [], 'learning_rates': []
        }

    def _setup_paths(self):
        domain_name = self.args.target_domain if self.args.dg_type == 'MDG' else self.args.source_domain
        variant_str = '' if self.args.remove_loss is None else f'_wo_{self.args.remove_loss}'
        base_name = f"{self.args.seg_model}_{self.args.backbone}_{self.args.dg_method}{variant_str}"

        self.save_weight_path = Path(
            self.args.data_dir) / domain_name / 'weight' / self.args.dg_type / f"{base_name}.pth"
        self.save_png_path = Path(self.args.data_dir) / domain_name / 'weight' / self.args.dg_type / f"{base_name}.png"
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)

    def _setup_model(self):
        self.model = MetaFD(self.args)
        self.model.to(self.device)

        self.optimizers['base'] = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=1e-4
        )
        self.optimizers['meta'] = optim.SGD(
            self.model.feature_disentangler.parameters(),
            lr=self.args.learning_rate,
            weight_decay=1e-4
        )

    def _get_loss_trackers(self) -> Tuple[Dict, Dict, Dict]:
        base_loss_details = {'seg_loss': 0, 'vae_loss': 0}
        mtr_loss_details = {'seg_loss': 0, 'sd_loss': 0, 'dd_loss': 0}
        mte_loss_details = {'seg_loss': 0, 'cyc_loss': 0, 'sd_loss': 0, 'dd_loss': 0}

        if self.args.remove_loss == 'lvd':
            mtr_loss_details.pop('dd_loss', None)
            mte_loss_details.pop('dd_loss', None)
        elif self.args.remove_loss == 'lsg':
            mtr_loss_details.pop('sd_loss', None)
            mte_loss_details.pop('sd_loss', None)
        elif self.args.remove_loss == 'lcon':
            base_loss_details.pop('vae_loss', None)
            mte_loss_details.pop('cyc_loss', None)

        return base_loss_details, mtr_loss_details, mte_loss_details

    def _reset_loss_trackers(self, trackers: Tuple[Dict, Dict, Dict]) -> Tuple[Dict, Dict, Dict]:
        base, mtr, mte = trackers
        for tracker in [base, mtr, mte]:
            for key in tracker:
                tracker[key] = 0
        return base, mtr, mte

    def _meta_learning_step(self, train_loader, train_step, base_loss_details, mtr_loss_details, mte_loss_details):
        base_update_loss, meta_train_loss, meta_test_loss = 0, 0, 0
        train_num = 0

        for _ in range(train_step):
            # Base model update
            b_x, b_y, b_d = next(iter(train_loader))
            b_x, b_y, b_d = self._to_device(b_x.float(), b_y.long(), b_d.long())

            base_loss, base_loss_dict = self.model(b_x, b_y, mode='base_training')
            self.optimizers['base'].zero_grad()
            base_loss.backward()
            self.optimizers['base'].step()

            # Meta-train
            mtr_x, mtr_y, mtr_d = next(iter(train_loader))
            mtr_x, mtr_y, mtr_d = self._to_device(mtr_x.float(), mtr_y.long(), mtr_d.long())

            mtr_loss, mtr_loss_dict = self.model(mtr_x, mtr_y, mode='meta-train')
            mtr_loss.requires_grad_(True)

            if self.args.remove_loss == 'l2l':
                base_params = list(self.model.parameters())
                mtr_grads = torch.autograd.grad(mtr_loss, base_params, create_graph=True, allow_unused=True)
                mtr_params = []
                for param, grad in zip(base_params, mtr_grads):
                    if grad is None:
                        grad = torch.zeros_like(param)
                    mtr_params.append((param - self.args.learning_rate * grad).requires_grad_())
            else:
                self.optimizers['meta'].zero_grad()
                mtr_loss.backward()
                self.optimizers['meta'].step()

            # Meta-test
            mte_x, mte_y, mte_d = next(iter(train_loader))
            mte_x, mte_y, mte_d = self._to_device(mte_x.float(), mte_y.long(), mte_d.long())

            if self.args.remove_loss == 'l2l':
                mte_loss, mte_loss_dict = ProtoModule(self.model).functional(mtr_params, True)(
                    mte_x, mte_y, mode='meta-test'
                )
                loss = mtr_loss + mte_loss
            else:
                mte_loss, mte_loss_dict = self.model(mte_x.float(), mte_y.long(), mode='meta-test')
                loss = mte_loss

            loss.requires_grad_(True)
            self.optimizers['meta'].zero_grad()
            loss.backward()
            self.optimizers['meta'].step()

            batch_size = len(b_y)
            self._update_loss_stats(base_loss_details, base_loss_dict, batch_size)
            self._update_loss_stats(mtr_loss_details, mtr_loss_dict, batch_size)
            self._update_loss_stats(mte_loss_details, mte_loss_dict, batch_size)

            total_loss = ((base_loss + loss) / 3).item() * batch_size
            base_update_loss += base_loss.item() * batch_size
            meta_train_loss += mtr_loss.item() * batch_size
            meta_test_loss += mte_loss.item() * batch_size
            train_num += batch_size

        return base_update_loss, meta_train_loss, meta_test_loss, train_num

    def _to_device(self, *tensors):
        return [tensor.to(self.device) for tensor in tensors]

    def _update_loss_stats(self, tracker: Dict, loss_dict: Dict, batch_size: int):
        for key, value in loss_dict.items():
            if key in tracker:
                tracker[key] += value * batch_size

    def _validate(self, val_loader) -> float:
        self.model.eval()
        val_loss, val_num = 0, 0

        with torch.no_grad():
            for step, (b_x, b_y, _) in enumerate(val_loader):
                b_x, b_y = self._to_device(b_x.float(), b_y.long())
                _, loss = self.model(b_x, b_y, mode='predict')
                val_loss += loss.item() * len(b_y)
                val_num += len(b_y)

        return val_loss / val_num if val_num > 0 else float('inf')

    def _print_epoch_summary(self, epoch: int, train_loss: float, val_loss: float,
                             base_loss_details: Dict, mtr_loss_details: Dict, mte_loss_details: Dict,
                             epoch_time: float):
        print(f"Epoch {epoch + 1}/{self.args.epoch}")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Base Loss Details: {base_loss_details}")
        print(f"   MTR Loss Details: {mtr_loss_details}")
        print(f"   MTE Loss Details: {mte_loss_details}")
        print(f"   Time: {epoch_time:.1f}s")
        print("-" * 60)

    def run(self) -> Tuple[torch.nn.Module, pd.DataFrame]:
        print("Starting MetaFD Training...")

        self._setup_model()

        train_loader, val_loader = dg_dataloader(self.args)
        train_step = len(train_loader)
        start_time = time.time()

        for epoch in range(self.args.epoch):
            epoch_start = time.time()
            self.model.train()
            base_loss_details, mtr_loss_details, mte_loss_details = self._get_loss_trackers()

            base_update_loss, meta_train_loss, meta_test_loss, train_num = self._meta_learning_step(
                train_loader, train_step, base_loss_details, mtr_loss_details, mte_loss_details
            )

            train_loss = (base_update_loss + meta_train_loss + meta_test_loss) / (3 * train_num)
            self.history['train_loss'].append(train_loss)

            for tracker in [base_loss_details, mtr_loss_details, mte_loss_details]:
                for key in tracker:
                    tracker[key] = round(tracker[key] / train_num, 4)

            val_loss = self._validate(val_loader)
            self.history['val_loss'].append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = self.model.state_dict().copy()

            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)

            self._print_epoch_summary(
                epoch, train_loss, val_loss,
                base_loss_details, mtr_loss_details, mte_loss_details,
                epoch_time
            )

        total_time = time.time() - start_time
        print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")

        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.model, self.save_weight_path)
        train_process = pd.DataFrame({
            "epoch": range(self.args.epoch),
            "train_loss_all": self.history['train_loss'],
            "val_loss_all": self.history['val_loss']
        })

        return self.model, train_process

    def plot_training_curve(self, train_process: pd.DataFrame):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(train_process.epoch, train_process.train_loss_all, 'b-', label="Train Loss", linewidth=2)
        plt.plot(train_process.epoch, train_process.val_loss_all, 'r-', label="Val Loss", linewidth=2)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_png_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MetaFD Training Script')

    parser.add_argument('--gpu_number', type=int, default=0, help='GPU device number')

    parser.add_argument('--data_dir', type=str, default="data/", help='dataset path')
    parser.add_argument('--dad', type=str, default="DABR", help='domain-agnostic dataset')
    parser.add_argument('--dg_type', type=str, default="MDG", help='MDG or SDG')
    parser.add_argument('--target_domain', type=str, default='CITY-OSM/CO-PA',
                        help='target domain in MDG setting')
    parser.add_argument('--source_domain', type=str, default='CITY-OSM/CO-PA',
                        help='source domain in SDG setting')

    parser.add_argument('--dg_method', type=str, default='MetaFD', help='domain generalization method')
    parser.add_argument('--seg_model', type=str, default='UNet', help='segmentation model')
    parser.add_argument('--backbone', type=str, default='vgg16', help='backbone network')

    parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--channel', type=int, default=3, help='input channels')
    parser.add_argument('--classes', type=int, default=3, help='number of classes')
    parser.add_argument('--crop_size', type=int, default=512, help='crop size')
    parser.add_argument('--output_stride', type=int, default=16, help='output stride')
    parser.add_argument('--f_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dimension')

    parser.add_argument('--seg_loss_factor', type=float, default=1.0)
    parser.add_argument('--sd_loss_factor', type=float, default=1.0)
    parser.add_argument('--dd_loss_factor', type=float, default=1.0)
    parser.add_argument('--cyc_loss_factor', type=float, default=1.0)

    parser.add_argument('--remove_loss', type=str, default=None,
                        help='ablation study: lvd, lsg, lcon, l2l')
    parser.add_argument('--l2l', type=bool, default=True,
                        help='whether using metalearning strategy')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    try:
        trainer = MetaFDTrainer(args)
        model, train_process = trainer.run()
        trainer.plot_training_curve(train_process)

    except Exception as e:
        print(f"Training failed: {e}")
        raise
