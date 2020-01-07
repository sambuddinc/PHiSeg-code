import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_code.utils import crossEntropy2D


class IncreaseResolution(nn.Module):
    def __init__(self, times, in_channels, out_channels):
        super().__init__()
        net = nn.Sequential()
        for i in range(times):
            if i != 0:
                in_channels = out_channels
            seq = nn.Sequential()
            seq.add_module('ups_%d' % i, nn.UpsamplingBilinear2d(scale_factor=2))
            seq.add_module('z%d_post' % i, Conv2dSame(in_channels, out_channels, 3))
            net.add_module('seq_%d' % i, seq)
        self.net = net

    def forward(self, x):
        return self.net(x)


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class phiseg_likelihood(nn.Module):
    def __init__(self, z_list, image_size, n_classes, config, device):
        super(phiseg_likelihood, self).__init__()
        n0 = config['n0']
        num_channels = [n0, 2*n0, 4*n0, 6*n0, 6*n0, 6*n0, 6*n0]

        latent_levels = config['latent_levels']
        resolution_levels = config['resolution_levels']
        self.image_size = image_size

        self.latent_levels = latent_levels
        self.resolution_levels = resolution_levels
        self.lvl_diff = resolution_levels - latent_levels

        post_z = [None] * latent_levels
        post_c_pre = [None] * latent_levels
        post_c_post = [None] * latent_levels
        s = [None] * latent_levels

        # Generate post_z
        for i in range(latent_levels):
            in_channels = 2
            seq = nn.Sequential()
            seq.add_module('z%d_post_1' % i, Conv2dSame(in_channels, num_channels[i], 3))
            seq.add_module('z%d_post_1r' % i, nn.ReLU(inplace=True))
            seq.add_module('z%d_post_lb' % i, nn.BatchNorm2d(num_channels[i]))
            seq.add_module('z%d_post_2' % i, Conv2dSame(num_channels[i], num_channels[i], 3))
            seq.add_module('z%d_post_2r' % i, nn.ReLU(inplace=True))
            seq.add_module('z%d_post_2b' % i, nn.BatchNorm2d(num_channels[i]))
            seq.add_module('z%d_ups' % i, IncreaseResolution(resolution_levels - latent_levels, num_channels[i], num_channels[i]))
            post_z[i] = seq.to(device)

        self.post_z = nn.ModuleList(post_z)

        # Upstream path
        post_c_pre[latent_levels - 1] = post_z[latent_levels - 1]
        post_c_post[latent_levels - 1] = post_z[latent_levels - 1]

        for i in reversed(range(latent_levels - 1)):
            seq = nn.Sequential()
            seq.add_module('post_z%d_ups' % i, nn.UpsamplingBilinear2d(scale_factor=2))
            seq.add_module('post_z%d_ups_c' % int(i+1), Conv2dSame(192, num_channels[i], 3)) # TODO: remove magic number here
            seq.add_module('z%d_post_1r' % i, nn.ReLU(inplace=True))
            seq.add_module('z%d_post_1b' % i, nn.BatchNorm2d(num_channels[i]))

            post_c_pre[i] = seq.to(device)

            seq = nn.Sequential()
            seq.add_module('post_c_%d_1' % i, Conv2dSame(num_channels[i] + num_channels[i], num_channels[i+self.lvl_diff], 3))
            seq.add_module('post_c_%d_r1' % i, nn.ReLU(inplace=True))
            seq.add_module('post_c_%d_b1' % i, nn.BatchNorm2d(num_channels[i+self.lvl_diff]))

            seq.add_module('post_c_%d_2' % i, Conv2dSame(num_channels[i+self.lvl_diff], num_channels[i+self.lvl_diff], 3))
            seq.add_module('post_c_%d_r2' % i, nn.ReLU(inplace=True))
            seq.add_module('post_c_%d_b2' % i, nn.BatchNorm2d(num_channels[i+self.lvl_diff]))

            post_c_post[i] = seq.to(device)

        self.post_c_pre = nn.ModuleList(post_c_pre)
        self.post_c_post = nn.ModuleList(post_c_post)

        for i in range(latent_levels):
            seq = nn.Sequential()
            seq.add_module('y_lvl%d' % i, nn.Conv2d(num_channels[i+self.lvl_diff], n_classes, 1))

            s[i] = seq.to(device)

        self.s = nn.ModuleList(s)
        self.z_list = z_list

    def forward(self, x, z_list):
        self.z_list = z_list

        post_z_holder = [None] * self.latent_levels
        post_c_pre_holder = [None] * self.latent_levels
        post_c_post_holder = [None] * self.latent_levels
        post_c_holder = [None] * self.latent_levels
        s_holder = [None] * self.latent_levels

        for i, layer in enumerate(self.post_z):
            post_z_holder[i] = layer(self.z_list[i])

        post_c_pre_holder[self.latent_levels - 1] = post_z_holder[self.latent_levels - 1]
        post_c_post_holder[self.latent_levels - 1] = post_z_holder[self.latent_levels - 1]
        post_c_holder[self.latent_levels - 1] = post_z_holder[self.latent_levels - 1]

        for i in reversed(range(self.latent_levels - 1)):
            pre_layer = self.post_c_pre[i]
            post_layer = self.post_c_post[i]
            pre = pre_layer(post_c_holder[i + 1])
            concat = torch.cat([post_z_holder[i], pre], dim=1)
            post_c_holder[i] = post_layer(concat)

        for i in range(self.latent_levels):
            s_layer = self.s[i]
            s = s_layer(post_c_holder[i])
            s_holder[i] = nn.functional.interpolate(s, size=self.image_size[0:2])

        return s_holder


class phiseg_prior(nn.Module):
    def __init__(self, z_list, zdim_0, config, device, generation_mode=False):
        super(phiseg_prior, self).__init__()
        n0 = config['n0']
        num_channels = [n0, 2*n0, 4*n0, 6*n0, 6*n0, 6*n0, 6*n0]

        latent_levels = config['latent_levels']
        resolution_levels = config['resolution_levels']

        self.latent_levels = latent_levels
        self.resolution_levels = resolution_levels
        self.generation_mode = generation_mode
        self.z_list = z_list

        pre_z = [None] * resolution_levels
        mu = [None] * latent_levels
        sigma = [None] * latent_levels
        z_inputs = [None] * latent_levels

        z_ups_mat = []
        for i in range(latent_levels):
            z_ups_mat.append([None]*latent_levels) # Encoding [original resolution][upsampled to]

        # Generate pre_z's
        for i in range(resolution_levels):
            pre_z_layers = nn.Sequential()
            if i != 0:
                pre_z_layers.add_module('avgpool2d', nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)))
                in_channels = num_channels[i-1]
            else:
                in_channels = 1

            pre_z_layers.add_module('z%d_pre_1c' % i, Conv2dSame(in_channels, num_channels[i], 3))
            pre_z_layers.add_module('z%d_pre_1r' % i, nn.ReLU(inplace=True))
            pre_z_layers.add_module('z%d_pre_1b' % i, nn.BatchNorm2d(num_channels[i]))

            pre_z_layers.add_module('z%d_pre_2c' % i, Conv2dSame(num_channels[i], num_channels[i], 3))
            pre_z_layers.add_module('z%d_pre_2r' % i, nn.ReLU(inplace=True))
            pre_z_layers.add_module('z%d_pre_2b' % i, nn.BatchNorm2d(num_channels[i]))

            pre_z_layers.add_module('z%d_pre_3c' % i, Conv2dSame(num_channels[i], num_channels[i], 3))
            pre_z_layers.add_module('z%d_pre_3r' % i, nn.ReLU(inplace=True))
            pre_z_layers.add_module('z%d_pre_3b' % i, nn.BatchNorm2d(num_channels[i]))

            pre_z[i] = pre_z_layers.to(device)

        self.pre_z = nn.ModuleList(pre_z)

        # Generate z's
        for i in reversed(range(latent_levels)):

            if i == latent_levels - 1:
                mu[i] = nn.Sequential(
                    Conv2dSame(num_channels[i+resolution_levels-latent_levels], zdim_0, 3),
                    nn.Identity(),
                    nn.Identity()).to(device)

                sigma[i] = nn.Sequential(
                    Conv2dSame(num_channels[i+resolution_levels-latent_levels], zdim_0, 1),
                    nn.Softplus(),
                    nn.Identity()).to(device)

            else:

                for j in reversed(range(0, i+1)):
                    if j == i:
                        in_channels = 2
                    else:
                        in_channels = zdim_0 * n0

                    z_below_ups = nn.Sequential()
                    z_below_ups.add_module('up_%d_%d' % (j, i), nn.UpsamplingBilinear2d(scale_factor=2))

                    z_below_ups.add_module('z%d_pre_1c_%d' % ((i+1),(j+1)), Conv2dSame(in_channels, zdim_0*n0, 3))
                    z_below_ups.add_module('z%d_pre_1r_%d' % ((i+1),(j+1)), nn.ReLU(inplace=True))
                    z_below_ups.add_module('z%d_pre_1b_%d' % ((i+1),(j+1)), nn.BatchNorm2d(zdim_0*n0))

                    z_below_ups.add_module('z%d_pre_2c_%d' % ((i+1),(j+1)), Conv2dSame(zdim_0*n0, zdim_0*n0, 3))
                    z_below_ups.add_module('z%d_pre_2r_%d' % ((i+1),(j+1)), nn.ReLU(inplace=True))
                    z_below_ups.add_module('z%d_pre_2b_%d' % ((i+1),(j+1)), nn.BatchNorm2d(zdim_0*n0))

                    z_ups_mat[j][i + 1] = z_below_ups.to(device)

                z_inputs[i] = nn.Sequential(
                    Conv2dSame(num_channels[i+resolution_levels-latent_levels] + zdim_0*n0, num_channels[i], 3),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(num_channels[i]),
                    Conv2dSame(num_channels[i], num_channels[i], 3),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(num_channels[i])).to(device)

                mu[i] = nn.Sequential(
                    nn.Conv2d(num_channels[i], zdim_0, 1),
                    nn.Identity(),
                    nn.Identity()).to(device)

                sigma[i] = nn.Sequential(
                    nn.Conv2d(num_channels[i], zdim_0, 1),
                    nn.Softplus(),
                    nn.Identity()).to(device)

        self.mu = nn.ModuleList(mu)
        self.sigma = nn.ModuleList(sigma)
        self.z_inputs = nn.ModuleList(z_inputs)
        z_ups_mat_ml_hold = nn.ModuleList()
        for i, row in enumerate(z_ups_mat):
            # z_ups_row_hold
            for i, col in enumerate(row):
                if col is None:
                    col = nn.Identity()
                row[i] = col
            row = nn.ModuleList(row)
            z_ups_mat_ml_hold.append(row)
        self.z_ups_mat = z_ups_mat_ml_hold

    def forward(self, x, z_list, generation_mode=False):
        self.z_list = z_list

        mu_sig_in_holder = [None] * self.resolution_levels

        for i, layer in enumerate(self.pre_z):
            x = layer(x)
            mu_sig_in_holder[i] = x

        mu_holder = [None] * self.latent_levels
        sigma_holder = [None] * self.latent_levels
        z_holder = [None] * self.latent_levels
        z_ups_mat_holder = []
        for i in range(self.latent_levels):
            z_ups_mat_holder.append([None] * self.latent_levels)

        for i in reversed(range(self.latent_levels)):

            if i == self.latent_levels - 1:
                mu_layer = self.mu[i]
                mu_holder[i] = mu_layer(mu_sig_in_holder[i+self.resolution_levels-self.latent_levels])
                sigma_layer = self.sigma[i]
                sigma_holder[i] = sigma_layer(mu_sig_in_holder[i+self.resolution_levels-self.latent_levels])
                z_holder[i] = mu_holder[i] + sigma_holder[i] * torch.randn_like(mu_holder[i])

            else:

                for j in reversed(range(0, i + 1)):
                    z_ups_mat_layer = self.z_ups_mat[j][i + 1]
                    z_ups_mat_holder[j][i + 1] = z_ups_mat_layer(z_ups_mat_holder[j+1][i+1])

                z_input = torch.cat([mu_sig_in_holder[i+self.resolution_levels-self.latent_levels], z_ups_mat_holder[i][i+1]], dim=1)
                z_input_layer = self.z_inputs[i]
                z_input = z_input_layer(z_input)

                mu_layer = self.mu[i]
                mu_holder[i] = mu_layer(z_input)
                sigma_layer = self.sigma[i]
                sigma_holder[i] = sigma_layer(z_input)
                z_holder[i] = mu_holder[i] + sigma_holder[i] * torch.randn_like(mu_holder[i])

            # if self.training:
            if not generation_mode:
                # use posterior samples
                z_ups_mat_holder[i][i] = self.z_list[i]
            else:
                # use prior samples
                z_ups_mat_holder[i][i] = z_holder[i]

        return z_holder, mu_holder, sigma_holder


class phiseg_posterior(nn.Module):
    def __init__(self, s_oh, zdim_0, config, device):
        super(phiseg_posterior, self).__init__()
        n0 = config['n0']
        num_channels = [n0, 2 * n0, 4 * n0, 6 * n0, 6 * n0, 6 * n0, 6 * n0]

        latent_levels = config['latent_levels']
        resolution_levels = config['resolution_levels']

        self.latent_levels = latent_levels
        self.resolution_levels = resolution_levels
        self.s_oh = s_oh

        pre_z = [None] * resolution_levels
        mu = [None] * latent_levels
        sigma = [None] * latent_levels
        z_inputs = [None] * latent_levels

        z_ups_mat = []
        for i in range(latent_levels):
            z_ups_mat.append([None] * latent_levels)

        for i in range(resolution_levels):
            pre_z_layers = nn.Sequential()
            if i != 0:
                pre_z_layers.add_module('avgpool2d', nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)))
                in_channels = num_channels[i-1]
            else:
                in_channels = 1 + config['nlabels']

            pre_z_layers.add_module('z%d_pre_1c' % i, Conv2dSame(in_channels, num_channels[i], 3))
            pre_z_layers.add_module('z%d_pre_1r' % i, nn.ReLU(inplace=True))
            pre_z_layers.add_module('z%d_pre_1b' % i, nn.BatchNorm2d(num_channels[i]))

            pre_z_layers.add_module('z%d_pre_2c' % i, Conv2dSame(num_channels[i], num_channels[i], 3))
            pre_z_layers.add_module('z%d_pre_2r' % i, nn.ReLU(inplace=True))
            pre_z_layers.add_module('z%d_pre_2b' % i, nn.BatchNorm2d(num_channels[i]))

            pre_z_layers.add_module('z%d_pre_3c' % i, Conv2dSame(num_channels[i], num_channels[i], 3))
            pre_z_layers.add_module('z%d_pre_3r' % i, nn.ReLU(inplace=True))
            pre_z_layers.add_module('z%d_pre_3b' % i, nn.BatchNorm2d(num_channels[i]))

            pre_z[i] = pre_z_layers.to(device)

        self.pre_z = nn.ModuleList(pre_z)

        # Generate z's
        for i in reversed(range(latent_levels)):

            if i == latent_levels - 1:
                mu[i] = nn.Sequential(
                    Conv2dSame(num_channels[i + resolution_levels - latent_levels], zdim_0, 3),
                    nn.Identity(),
                    nn.Identity()).to(device)

                sigma[i] = nn.Sequential(
                    Conv2dSame(num_channels[i + resolution_levels - latent_levels], zdim_0, 1),
                    nn.Softplus(),
                    nn.Identity()).to(device)

            else:

                for j in reversed(range(0, i + 1)):
                    if j == i:
                        in_channels = 2
                    else:
                        in_channels = zdim_0 * n0

                    z_below_ups = nn.Sequential()
                    z_below_ups.add_module('up_%d_%d' % (j, i), nn.UpsamplingBilinear2d(scale_factor=2))

                    z_below_ups.add_module('z%d_up_1c_%d' % ((i+1),(j+1)), Conv2dSame(in_channels, zdim_0 * n0, 3))
                    z_below_ups.add_module('z%d_up_1r_%d' % ((i+1),(j+1)), nn.ReLU(inplace=True))
                    z_below_ups.add_module('z%d_up_1b_%d' % ((i+1),(j+1)), nn.BatchNorm2d(zdim_0 * n0))

                    z_below_ups.add_module('z%d_up_2c_%d' % ((i+1),(j+1)), Conv2dSame(zdim_0 * n0, zdim_0 * n0, 3))
                    z_below_ups.add_module('z%d_up_2r_%d' % ((i+1),(j+1)), nn.ReLU(inplace=True))
                    z_below_ups.add_module('z%d_up_2b_%d' % ((i+1),(j+1)), nn.BatchNorm2d(zdim_0 * n0))

                    z_ups_mat[j][i + 1] = z_below_ups.to(device)
                    # z_ups_mat_ml[j][i+1] = z_below_ups.to(device)

                z_inputs[i] = nn.Sequential(
                    Conv2dSame(num_channels[i + resolution_levels - latent_levels] + zdim_0 * n0, num_channels[i], 3),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(num_channels[i]),
                    Conv2dSame(num_channels[i], num_channels[i], 3),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(num_channels[i])).to(device)

                mu[i] = nn.Sequential(
                    Conv2dSame(num_channels[i], zdim_0, 1),
                    nn.Identity(),
                    nn.Identity()).to(device)

                sigma[i] = nn.Sequential(
                    Conv2dSame(num_channels[i], zdim_0, 1),
                    nn.Softplus(),
                    nn.Identity()).to(device)

        self.mu = nn.ModuleList(mu)
        self.sigma = nn.ModuleList(sigma)
        self.z_inputs = nn.ModuleList(z_inputs)
        z_ups_mat_ml_hold = nn.ModuleList()
        for i, row in enumerate(z_ups_mat):
            for i, col in enumerate(row):
                if col is None:
                    col = nn.Identity()
                row[i] = col
            row = nn.ModuleList(row)
            z_ups_mat_ml_hold.append(row)
        self.z_ups_mat = z_ups_mat_ml_hold

    def forward(self, x, target):

        mu_sig_in_holder = [None] * self.resolution_levels

        for i, layer in enumerate(self.pre_z):
            if i == 0:
                x = torch.cat([x, target-0.5], dim=1)
                x = layer(x)
                mu_sig_in_holder[i] = x
            else:
                x = layer(x)
                mu_sig_in_holder[i] = x

        mu_holder = [None] * self.latent_levels
        sigma_holder = [None] * self.latent_levels
        z_holder = [None] * self.latent_levels
        z_ups_mat_holder = []
        for i in range(self.latent_levels):
            z_ups_mat_holder.append([None] * self.latent_levels)

        for i in reversed(range(self.latent_levels)):
            if i == self.latent_levels - 1:
                mu_layer = self.mu[i]
                mu_holder[i] = mu_layer(mu_sig_in_holder[i+self.resolution_levels-self.latent_levels])
                sigma_layer = self.sigma[i]
                sigma_holder[i] = sigma_layer(mu_sig_in_holder[i+self.resolution_levels-self.latent_levels])

                z_holder[i] = mu_holder[i] + (sigma_holder[i] * torch.randn_like(mu_holder[i]))

            else:

                for j in reversed(range(0, i + 1)):
                    z_ups_mat_layer = self.z_ups_mat[j][i + 1]
                    z_ups_mat_holder[j][i + 1] = z_ups_mat_layer(z_ups_mat_holder[j+1][i+1])

                z_input = torch.cat([mu_sig_in_holder[i+self.resolution_levels-self.latent_levels], z_ups_mat_holder[i][i+1]], dim=1)
                z_input_layer = self.z_inputs[i]
                z_input = z_input_layer(z_input)

                mu_layer = self.mu[i]
                mu_holder[i] = mu_layer(z_input)
                sigma_layer = self.sigma[i]
                sigma_holder[i] = sigma_layer(z_input)
                z_holder[i] = mu_holder[i] + (sigma_holder[i] * torch.randn_like(mu_holder[i]))

            z_ups_mat_holder[i][i] = z_holder[i]

        return z_holder, mu_holder, sigma_holder


class PhiSeg(nn.Module):
    def __init__(self, config, device):
        super(PhiSeg, self).__init__()
        s_oh = torch.arange(0, config['nlabels'])
        s_oh = F.one_hot(s_oh, config['nlabels'])
        s_oh = s_oh.to(device)
        s_oh = s_oh.float()
        self.config = config

        phi_post = phiseg_posterior(s_oh, config['zdim_0'], config, device)
        self.phi_post = phi_post

        z_list = [None] * config['latent_levels']
        self.phi_prior = phiseg_prior(z_list, config['zdim_0'], config, device)

        self.phi_like = phiseg_likelihood(z_list, config['image_size'], config['nlabels'], config, device)

    def forward(self, x, target, mode='train'):

        if mode == 'train':

            s_oh = F.one_hot(target, self.config['nlabels'])
            s_oh = s_oh.float()
            s_oh = s_oh.squeeze().permute(0,3,1,2)

            z_list, mu_list, sig_list = self.phi_post.forward(x, s_oh)
            prior_z_list, prior_mu_list, prior_sig_list = self.phi_prior.forward(x, z_list)
            s_out_list = self.phi_like.forward(x, z_list)

            s_out_sm_list = [None] * self.config['latent_levels']
            for ii in range(self.config['latent_levels']):
                s_out_sm_list[ii] = F.softmax(s_out_list[ii], dim=1)

            s_out = self._aggregate_output_list(s_out_list, use_softmax=False)

            loss_dict = {}
            loss_tot = 0

            loss_dict, loss_tot = self.add_residual_multinoulli_loss(loss_dict, loss_tot, s_out_list, target)

            loss_dict, loss_tot = self.add_heirarchical_KL_div_loss(loss_dict, loss_tot, prior_mu_list, prior_sig_list,
                                                                    mu_list, sig_list)

            segmentation = s_out

        else:
            z_list = []
            prior_z_list_gen, prior_mu_list_gen, prior_sig_list_gen = self.phi_prior.forward(x, z_list, generation_mode=True)
            s_out_eval_list = self.phi_like.forward(x, prior_z_list_gen)

            s_out_eval_sm_list = [None] * self.config['latent_levels']
            for ii in range(self.config['latent_levels']):
                s_out_eval_sm_list[ii] = F.softmax(s_out_eval_list[ii], dim=1)

            s_out_eval = self._aggregate_output_list(s_out_eval_list, use_softmax=False)

            eval_ce = crossEntropy2D(s_out_eval, target)

            loss_dict = {'eval_ce': eval_ce}
            loss_tot = eval_ce

            segmentation = s_out_eval

        return loss_tot, loss_dict, segmentation


    def _aggregate_output_list(self, output_list, use_softmax=True):
        s_accum = output_list[-1]
        for i in range(len(output_list) - 1):
            s_accum += output_list[i]
        if use_softmax:
            return F.softmax(s_accum)
        return s_accum

    def add_residual_multinoulli_loss(self, loss_dict, loss_tot, s_out_list, s_inp_oh):
        s_accum = [None] * self.config['latent_levels']

        for ii, s_ii in zip(reversed(range(self.config['latent_levels'])),
                            reversed(s_out_list)):
            if ii == self.config['latent_levels'] - 1:
                s_accum[ii] = s_ii
                loss_dict['residual_multinoulli_loss_lvl%d' % ii] = self.multinoulli_loss_with_logits(s_inp_oh, s_accum[ii])
            else:
                s_accum[ii] = s_accum[ii+1] + s_ii
                loss_dict['residual_multinoulli_loss_lvl%d' % ii] = self.multinoulli_loss_with_logits(s_inp_oh, s_accum[ii])

            # print('added re multi loss at level %d' % (ii))
            loss_tot += self.config['residual_multinoulli_loss_weight'] * loss_dict['residual_multinoulli_loss_lvl%d' % ii]

        return loss_dict, loss_tot


    def multinoulli_loss_with_logits(self, x_gt, y_target):
        ce = crossEntropy2D(y_target, x_gt) # TODO: double check multinoulli loss
        return ce

    def add_heirarchical_KL_div_loss(self, loss_dict, loss_tot, prior_mu_list, prior_sig_list, mu_list, sig_list):

        if self.config['exponential_weighting']:
            level_weights = [4**i for i in list(range(self.config['latent_levels']))]
        else:
            level_weights = [1]*self.config['latent_levels']

        for ii, mu_i, sig_i in zip(reversed(range(self.config['latent_levels'])),
                                   reversed(mu_list),
                                   reversed(sig_list)):
            loss_dict['KL_divergence_loss_lvl%d' % ii] = level_weights[ii] * self.KL_two_gauss_with_diag_cov(mu_i, sig_i, prior_mu_list[ii], prior_sig_list[ii])
            # print('added heir kl loss at level %d with alpha_%d=%d' % (ii, ii, level_weights[ii]))
            loss_tot += self.config['KL_divergence_loss_weight'] * loss_dict['KL_divergence_loss_lvl%d' % ii]

        return loss_dict, loss_tot

    def KL_two_gauss_with_diag_cov(self, mu0, sigma0, mu1, sigma1):

        sig0_fl = sigma0.flatten()
        sig1_fl = sigma1.flatten()

        sigma0_fs = torch.mul(sig0_fl, sig0_fl)
        sigma1_fs = torch.mul(sig1_fl, sig1_fl)

        logsigma0 = torch.log(sigma0_fs + 1e-10)
        logsigma1 = torch.log(sigma1_fs + 1e-10)

        mu0_f = mu0.flatten()
        mu1_f = mu1.flatten()

        sum_in = torch.div(sigma0_fs + torch.mul(mu1_f - mu0_f, mu1_f - mu0_f), sigma1_fs + 1e-10) + logsigma1 - logsigma0 - 1
        red_sum = 0.5 * sum_in
        out = torch.mean(red_sum)

        return out


