import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report

from network.model import MetaFD
from utils.cfgs import FILE_FORMAT, CLASS_NAME
from utils.data_reader import MyDataset, label2image


class MetaFDPredictor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_number}" if torch.cuda.is_available() else "cpu")
        self._setup_paths()
        self._load_model()
        self._setup_metrics()

    def _setup_paths(self):
        self.test_path = Path(self.args.data_dir) / self.args.td / 'sample' / 'val'
        weight_stem = Path(self.args.weight_path).stem
        self.result_path = Path(self.args.data_dir) / self.args.td / 'result' / self.args.dg_type / weight_stem
        self.result_path.mkdir(parents=True, exist_ok=True)

        self.dataset_name = self.args.td.split('/')[-1]

    def _load_model(self):
        self.model = MetaFD(self.args)
        self.model.to(self.device)
        self.model.eval()

        try:
            checkpoint = torch.load(self.args.weight_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
            else:
                self.model.load_state_dict(checkpoint.state_dict(), strict=False)
            print(f"Successfully loaded model from {self.args.weight_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _setup_metrics(self):
        self.metrics = {
            'f1': {},
            'precision': {},
            'recall': {}
        }

        for target_name in CLASS_NAME[self.dataset_name]:
            for metric in self.metrics:
                self.metrics[metric][target_name] = []

    def _create_dataloader(self):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_data = MyDataset(
            data_root=str(self.test_path),
            transform=test_transform,
            img_format=FILE_FORMAT[self.args.td],
            label_format=FILE_FORMAT[self.args.td],
            filename=True
        )

        return DataLoader(
            test_data,
            batch_size=self.args.bs,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )

    def _save_prediction(self, prediction, filename):
        img = Image.fromarray(np.uint8(label2image(prediction, self.dataset_name))).convert('RGB')
        save_path = self.result_path / Path(filename).name
        img.save(save_path)
        return save_path

    def _calculate_metrics(self, true_labels, pred_labels):
        true_flat = true_labels.flatten()
        pred_flat = pred_labels.flatten()

        report = classification_report(
            true_flat, pred_flat,
            target_names=CLASS_NAME[self.dataset_name],
            digits=4,
            labels=list(range(self.args.classes)),
            output_dict=True,
            zero_division=0
        )

        return report

    def _update_metrics(self, report):
        for target_name in CLASS_NAME[self.dataset_name]:
            if target_name in report and report[target_name]['support'] > 0:
                self.metrics['f1'][target_name].append(report[target_name]['f1-score'])
                self.metrics['precision'][target_name].append(report[target_name]['precision'])
                self.metrics['recall'][target_name].append(report[target_name]['recall'])

    def _save_metrics_report(self):
        report_path = self.result_path / 'classification_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")

            for target_name in CLASS_NAME[self.dataset_name]:
                if not self.metrics['f1'][target_name]:
                    continue

                f.write(f"Class: {target_name}\n")
                f.write(
                    f"  F1-score:    {np.mean(self.metrics['f1'][target_name]):.4f} ± {np.std(self.metrics['f1'][target_name]):.4f}\n")
                f.write(
                    f"  Precision:   {np.mean(self.metrics['precision'][target_name]):.4f} ± {np.std(self.metrics['precision'][target_name]):.4f}\n")
                f.write(
                    f"  Recall:      {np.mean(self.metrics['recall'][target_name]):.4f} ± {np.std(self.metrics['recall'][target_name]):.4f}\n")
                f.write(f"  Sample size: {len(self.metrics['f1'][target_name])}\n\n")

            # 添加宏观平均
            macro_f1 = np.mean([np.mean(scores) for scores in self.metrics['f1'].values() if scores])
            f.write(f"Macro Average F1-score: {macro_f1:.4f}\n")

        print(f"Metrics report saved to: {report_path}")

    def run(self):
        test_loader = self._create_dataloader()
        total_samples = len(test_loader.dataset)

        print(f"Starting prediction on {total_samples} samples...")
        print(f"Results will be saved to: {self.result_path}")

        self.model.eval()
        total_time = 0

        with torch.no_grad():
            for batch_idx, (batch_filenames, batch_images, batch_labels) in enumerate(test_loader):
                batch_start = time.time()
                batch_images = batch_images.float().to(self.device, non_blocking=True)
                batch_labels = batch_labels.long().to(self.device, non_blocking=True)
                outputs, _ = self.model(batch_images, batch_labels, mode='predict')
                outputs = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                batch_labels_np = batch_labels.cpu().numpy()
                predictions_np = predictions.cpu().numpy()

                for i in range(len(predictions_np)):
                    self._save_prediction(predictions_np[i], batch_filenames[i])
                    report = self._calculate_metrics(batch_labels_np[i], predictions_np[i])
                    self._update_metrics(report)

                batch_time = time.time() - batch_start
                total_time += batch_time
                processed_samples = (batch_idx + 1) * self.args.bs

                print(f'Batch {batch_idx + 1}/{len(test_loader)} | '
                      f'Progress: {processed_samples}/{total_samples} '
                      f'({100. * processed_samples / total_samples:.1f}%) | '
                      f'Time: {batch_time:.2f}s')
        self._save_metrics_report()

        print(f"Prediction completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"Macro F1-score: {np.mean([np.mean(scores) for scores in self.metrics['f1'].values() if scores]):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MetaFD Prediction Script')
    parser.add_argument('--gpu_number', type=int, default=1, help='GPU device number')
    parser.add_argument('--td', type=str, default='CITY-OSM/CO-PO', help='target domain')
    parser.add_argument('--data_dir', type=str, default="data/", help='dataset path')
    parser.add_argument('--bs', type=int, default=20, help='batch size')
    parser.add_argument('--dg_type', type=str, default="MDG", help='mdg or sdg')
    parser.add_argument('--seg_model', type=str, default='UNet', help='segmentation model')
    parser.add_argument('--backbone', type=str, default='vgg16', help='backbone network')
    parser.add_argument('--weight_path', type=str,
                        default='/home/lcb/MetaFD/data/CITY-OSM/CO-PA/weight/MDG/UNet_vgg16_MetaFD.pth',
                        help='path to model weights')
    parser.add_argument('--channel', type=int, default=3, help='input channels')
    parser.add_argument('--classes', type=int, default=3, help='number of classes')
    parser.add_argument('--f_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dimension')
    parser.add_argument('--output_stride', type=int, default=16, help='output stride')
    parser.add_argument('--seg_loss_factor', type=float, default=1.0)
    parser.add_argument('--sd_loss_factor', type=float, default=1.0)
    parser.add_argument('--dd_loss_factor', type=float, default=1.0)
    parser.add_argument('--cyc_loss_factor', type=float, default=1.0)
    parser.add_argument('--remove_loss', type=str, default=None,
                        help='ablation study: lvd, lsg, lcon, l2l')
    parser.add_argument('--l2l', type=bool, default=True,
                        help='whether using metalearning strategy')

    args = parser.parse_args()

    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Model weights not found: {args.weight_path}")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    try:
        predictor = MetaFDPredictor(args)
        predictor.run()
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise
