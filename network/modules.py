import torch.nn as nn
import torch
from torch.nn import functional as F
from network.backbone import VGG16, Xception, ASPP_v3, ResNet, CBAM


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=2):
        return input.view(input.size(0), -1, size, size)


class FeaEncoder(nn.Module):
    def __init__(self, args):
        super(FeaEncoder, self).__init__()
        self.args = args
        if args.seg_model == 'UNet':
            self.base_model = VGG16(args)
            self.layers = ["concat1", "concat2", "concat3", "concat4", "downsample4"]
            cn = [64, 128, 256, 512, 512]
            self.layers = ["concat1", "concat2", "concat3", "concat4", "downsample4"]
            self.relu = nn.ReLU(inplace=True)
            self.decoder_layer1 = nn.Sequential(
                nn.Conv2d(cn[3], cn[3] * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[3] * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(cn[3] * 2, cn[3] * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[3] * 2),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(cn[3] * 2, cn[3], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[3]),
                nn.ReLU(inplace=True)
            )
            self.decoder_layer2 = nn.Sequential(
                nn.Conv2d(cn[3] * 2, cn[3], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(cn[3], cn[3], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[3]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(cn[3], cn[2], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[2]),
                nn.ReLU(inplace=True)
            )
            self.decoder_layer3 = nn.Sequential(
                nn.Conv2d(cn[2] * 2, cn[2], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(cn[2], cn[2], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[2]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(cn[2], cn[1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[1]),
                nn.ReLU(inplace=True)
            )
            self.decoder_layer4 = nn.Sequential(
                nn.Conv2d(cn[1] * 2, cn[1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(cn[1], cn[1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(cn[1], cn[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[0]),
                nn.ReLU(inplace=True)
            )
            self.decoder_layer5 = nn.Sequential(
                nn.Conv2d(cn[0] * 2, args.f_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cn[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(args.f_dim, args.f_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(args.f_dim),
                nn.ReLU(inplace=True)
            )

        elif args.seg_model == 'DeepLabv3+':
            cn = [128, 256, 728, 1024, 2048]
            if args.output_stride == 32:
                rates = [3, 6, 9]
            if args.output_stride == 16:
                rates = [6, 12, 18]
            if args.output_stride == 8:
                rates = [12, 24, 36]
            self.backbone = Xception(args)
            self.ASPP = ASPP_v3(in_channels=cn[4], atrous_rates=rates)
            self.conv1 = nn.Sequential(
                nn.Conv2d(cn[1], 48, 1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Conv2d(256, args.f_dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(args.f_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
            self.layers = ["downsample1", "downsample2", "downsample3", "downsample4", 'downsample5']

        elif args.seg_model == 'SCAttNet':
            self.backbone = ResNet(args)
            cn = [64, 256, 512, 1024, 2048]
            self.conv1 = nn.Conv2d(cn[4], args.f_dim, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(args.f_dim)
            self.relu = nn.ReLU()
            self.CBAM = CBAM(planes=args.f_dim)
        else:
            raise Exception('error segmentation network!')

    def forward(self, x):
        if self.args.seg_model == 'UNet':
            output = {}
            for name, layer in self.base_model._modules.items():
                x = layer(x)
                if name in self.layers:
                    output[name] = x
            x1 = output['concat1']
            x2 = output['concat2']
            x3 = output['concat3']
            x4 = output['concat4']
            x5 = output['downsample4']
            score = self.decoder_layer1(x5)
            score = torch.cat([score, x4], dim=1)
            score = self.decoder_layer2(score)
            score = torch.cat([score, x3], dim=1)
            score = self.decoder_layer3(score)
            score = torch.cat([score, x2], dim=1)
            score = self.decoder_layer4(score)
            score = torch.cat([score, x1], dim=1)
            out = self.decoder_layer5(score)
            return out

        elif self.args.seg_model == 'DeepLabv3+':
            size = x.shape[-2:]
            output = {}
            for name, layer in self.backbone._modules.items():
                x = layer(x)
                if name in self.layers:
                    output[name] = x
            x2 = output['downsample2']
            x1 = output['downsample5']
            x1 = self.ASPP(x1)
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            x2 = self.conv1(x2)
            x = torch.cat((x1, x2), dim=1)
            x = self.conv2(x)
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            return x

        elif self.args.seg_model == 'SCAttNet':
            size = x.shape[-2:]
            x = self.backbone(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.CBAM(x)
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            return x



class TaskClassifier(nn.Module):
    def __init__(self, args):
        super(TaskClassifier, self).__init__()
        self.classifier = nn.Conv2d(args.f_dim, args.classes, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.classifier(x)
        return out


class Disentangler(nn.Module):
    def __init__(self, args):
        super(Disentangler, self).__init__()
        # reference: https://github.com/AmingWu/VDD-DAOD.git
        self.ds_disentangler = nn.Sequential(
            nn.Conv2d(args.f_dim, args.f_dim // 2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(args.f_dim // 2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(args.f_dim // 2, args.f_dim // 2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(args.f_dim // 2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(args.f_dim // 2, args.f_dim, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(args.f_dim),
        )

    def forward(self, x):
        return self.ds_disentangler(x)


class VAEEncoder(nn.Module):
    # reference: https://github.com/sksq96/pytorch-vae.git
    def __init__(self, args):
        super(VAEEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(args.f_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Flatten()
        )
        self.z_mean = nn.Linear(256 * 2 * 2, args.latent_dim)
        self.z_log_var = nn.Linear(256 * 2 * 2, args.latent_dim)

    def forward(self, x):
        x = self.network(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


class VAEDecoder(nn.Module):
    def __init__(self, args):
        super(VAEDecoder, self).__init__()
        self.fc = nn.Linear(args.latent_dim, 256 * 2 * 2)
        self.network = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, args.f_dim, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.network(x)
        return x
