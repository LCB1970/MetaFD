import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from network.modules import FeaEncoder, TaskClassifier, VAEDecoder, VAEEncoder, Disentangler


class EntropyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(EntropyLoss, self).__init__(size_average, reduce, reduction)

    # https://github.com/human-analysis/MaxEnt-ARL.git
    # input is probability distribution of output classes
    def forward(self, input):
        input = F.softmax(input, dim=1)
        if (input < 0).any() or (input > 1).any():
            raise Exception('Entropy Loss takes probabilities 0<=input<=1')

        input = input + 1e-16  # for numerical stability while taking log
        H = -torch.mean(torch.sum(input * torch.log(input), dim=1))

        return H


class MetaFD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.remove_loss = args.remove_loss
        self.fea_encoder = FeaEncoder(args)
        self.classifier = TaskClassifier(args)
        self.feature_disentangler = Disentangler(args)
        self.vae_encoder = VAEEncoder(args)
        self.vae_decoder = VAEDecoder(args)
        self.seg_criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.entropy = EntropyLoss()
        self.softplus = nn.Softplus()
        self.args = args

    @staticmethod
    def reparameterization(z_mean, z_log_var):
        """
        :param args:
        :return:
        """
        eps = torch.randn_like(z_log_var)
        z = z_mean + torch.exp(z_log_var / 2) * eps
        return z

    def vae_loss(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = self.mse_loss(outputs, inputs)
        kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
        kl_loss = -0.5 * torch.sum(kl_loss)
        vae_loss = torch.mean(reconstruction_loss + kl_loss)
        return vae_loss

    def semantic_guidance_loss(self, f_out, fdi_out, fds_out):
        return self.softplus(self.entropy(fdi_out) - self.entropy(f_out)) + self.softplus(
            self.entropy(f_out) - self.entropy(fds_out))

    def vector_decomposed_loss(self, fdi, fds):
        pdi = F.adaptive_avg_pool2d(fdi, (1, 1)).reshape((-1, fdi.size(1)))
        pds = F.adaptive_avg_pool2d(fds, (1, 1)).reshape((-1, fds.size(1)))
        M = torch.abs(torch.mm(F.normalize(pdi, 2, dim=1), F.normalize(pds, 2, dim=1).t()))
        ortho_loss = torch.mean(M)
        return ortho_loss

    def forward(self, x, y, mode='base_training'):
        if mode == 'base_training':
            fea = self.fea_encoder(x)
            with torch.no_grad():
                fea_ds = self.feature_disentangler(fea)
            y_out = self.classifier(fea - fea_ds)
            seg_loss = self.seg_criterion(y_out, y)

            if self.remove_loss == 'lcon':
                return seg_loss, {'seg_loss': seg_loss.item()}

            z_mean, z_var = self.vae_encoder(fea_ds)
            z = self.reparameterization(z_mean, z_var)
            fds_recon = self.vae_decoder(z)
            vae_loss = self.vae_loss(fds_recon, fea_ds, z_mean, z_var)

            base_loss = seg_loss + vae_loss
            return base_loss, {'seg_loss': seg_loss.item(), 'vae_loss': vae_loss.item()}

        elif mode == 'meta-train':
            with torch.no_grad():
                fea = self.fea_encoder(x)
            fea_ds = self.feature_disentangler(fea)
            with torch.no_grad():
                f_out = self.classifier(fea)
                fdi_out = self.classifier(fea - fea_ds)
                fds_out = self.classifier(fea_ds)
            sd_loss = self.semantic_guidance_loss(f_out, fdi_out, fds_out)
            dd_loss = self.vector_decomposed_loss(fea - fea_ds, fea_ds)
            seg_loss = self.seg_criterion(fdi_out, y)
            if self.remove_loss == 'lsg':
                mtr_loss = self.args.seg_loss_factor * seg_loss + self.args.dd_loss_factor * dd_loss
                return mtr_loss, {'seg_loss': seg_loss.item(), 'dd_loss': dd_loss.item()}
            elif self.remove_loss == 'lvd':
                mtr_loss = self.args.seg_loss_factor * seg_loss + self.args.sd_loss_factor * sd_loss
                return mtr_loss, {'seg_loss': seg_loss.item(), 'sd_loss': sd_loss.item()}
            else:
                mtr_loss = self.args.seg_loss_factor * seg_loss + self.args.dd_loss_factor * dd_loss + self.args.sd_loss_factor * sd_loss
                return mtr_loss, {'seg_loss': seg_loss.item(), 'sd_loss': sd_loss.item(), 'dd_loss': dd_loss.item()}

        elif mode == 'meta-test':
            with torch.no_grad():
                fea = self.fea_encoder(x)
            fea_ds = self.feature_disentangler(fea)
            fea_di = fea - fea_ds
            z_random = torch.randn((fea.size(0), self.args.latent_dim)).to(x.device)
            with torch.no_grad():
                fea_ds_generate = self.vae_decoder(z_random)

            fea_fake = fea_ds_generate + fea_di
            fea_ds_fake = self.feature_disentangler(fea_fake)
            fea_di_fake = fea_fake - fea_ds_fake

            with torch.no_grad():
                f_out = self.classifier(fea)
                fdi_out = self.classifier(fea_di)
                fds_out = self.classifier(fea_ds)
                f_out_fake = self.classifier(fea_fake)
                fdi_out_fake = self.classifier(fea_di_fake)
                fds_out_fake = self.classifier(fea_ds_fake)

            cyc_loss = (self.mse_loss(fea_ds_fake, fea_ds_generate) + self.mse_loss(fea_di_fake, fea_di)) / 2
            dd_loss = (self.vector_decomposed_loss(fea_di, fea_ds) + self.vector_decomposed_loss(fea_di_fake,
                                                                                                 fea_ds_fake)) / 2
            sd_loss = (self.semantic_guidance_loss(f_out, fdi_out, fds_out) + self.semantic_guidance_loss(
                f_out_fake, fdi_out_fake, fds_out_fake)) / 2
            seg_loss = (self.seg_criterion(fdi_out, y) + self.seg_criterion(fdi_out_fake, y)) / 2

            if self.remove_loss == 'lsg':
                mte_loss = self.args.seg_loss_factor * seg_loss + self.args.cyc_loss_factor * cyc_loss + self.args.dd_loss_factor * dd_loss
                return mte_loss, {'seg_loss': seg_loss.item(), 'cyc_loss': cyc_loss.item(), 'dd_loss': dd_loss.item()}
            elif self.remove_loss == 'lvd':
                mte_loss = self.args.seg_loss_factor * seg_loss + self.args.cyc_loss_factor * cyc_loss + self.args.sd_loss_factor * sd_loss
                return mte_loss, {'seg_loss': seg_loss.item(), 'cyc_loss': cyc_loss.item(), 'sd_loss': sd_loss.item()}
            elif self.remove_loss == 'lcon':
                mte_loss = self.args.seg_loss_factor * seg_loss + self.args.dd_loss_factor * dd_loss + self.args.sd_loss_factor * sd_loss
                return mte_loss, {'seg_loss': seg_loss.item(), 'sd_loss': sd_loss.item(), 'dd_loss': dd_loss.item()}
            else:
                mte_loss = (self.args.seg_loss_factor * seg_loss + self.args.cyc_loss_factor * cyc_loss
                            + self.args.dd_loss_factor * dd_loss + self.args.sd_loss_factor * sd_loss)
                return mte_loss, {'seg_loss': seg_loss.item(), 'cyc_loss': cyc_loss.item(),
                                  'sd_loss': sd_loss.item(), 'dd_loss': dd_loss.item()}
        elif mode == 'predict':
            fea = self.fea_encoder(x)
            with torch.no_grad():
                fea_ds = self.feature_disentangler(fea)
            y_out = self.classifier(fea - fea_ds)
            seg_loss = self.seg_criterion(y_out, y) if y is not None else None
            return y_out, seg_loss

