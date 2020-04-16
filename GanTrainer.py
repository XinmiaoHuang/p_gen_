import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from base.base_trainer import BaseTrainer
from loss.dloss import CriterionD
from loss.ganloss import GANLoss
from base.utils import make_parts_shape
import torch.nn as nn


class GanTrainer(BaseTrainer):
    def __init__(self, opt, dataloader, criterion, model, testloader):
        super().__init__(opt, dataloader, criterion, model.gnet)
        self.testloader = testloader
        self.dnet = model.dnet
        self.criterion_d = CriterionD()
        self.criterionGAN = GANLoss(use_lsgan=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train(self):
        print(f'Use device: {self.device}'
              f'Network:\n'
              f'In channels: {self.n_channels}  '
              f'Out channels: {self.n_classes}')


        self.model.to(device=self.device)
        self.dnet.to(device=self.device)

        optimizer_G = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer_D = torch.optim.Adam(self.dnet.parameters(), lr=self.learning_rate)

        test_batch = iter(self.testloader).next()
        test_img = test_batch['input']['s_map']
        test_img = test_img.type(torch.FloatTensor).to(self.device)
        test_pose = test_batch['input']['t_pose']
        test_pose = test_pose.type(torch.FloatTensor).to(self.device)
        for epoch in range(self.epochs):
            for idx, data_batch in enumerate(self.dataloader):
                img = data_batch['input']['s_map']
                t_pose = data_batch['input']['t_pose']
                target = data_batch['input']['t_map']
                img = img.type(torch.FloatTensor).to(self.device)
                t_pose = t_pose.type(torch.FloatTensor).to(self.device)
                target = target.type(torch.FloatTensor).to(self.device)

                self.set_requires_grad(self.dnet, True)
                self.dnet.zero_grad()
                real_out_d, mid_real = self.dnet(target)
                label_d = torch.full(real_out_d.size(), 1, device=self.device)
                real_loss_d = self.criterion_d(real_out_d, label_d)
                real_loss_d.backward()

                fake = self.model(img, t_pose)
                fake_out_d, _ = self.dnet(fake.detach())
                label_d.fill_(0)
                fake_loss_d = self.criterion_d(fake_out_d, label_d)
                fake_loss_d.backward()

                err_d = real_loss_d + fake_loss_d
                optimizer_D.step()

                self.set_requires_grad(self.dnet, False)
                self.model.zero_grad()
                fake_score_d, mid_fake = self.dnet(fake)
                label_d.fill_(1)
                l_gan1 = self.criterionGAN(fake_score_d, label_d)

                # use fm loss
                loss_fm = 0
                for i in range(len(mid_fake)):
                    loss_fm += nn.L1Loss()(mid_fake[i], mid_real[i].detach())
                #

                loss_l1, loss_p = self.criterionPerPixel(fake, target)
                err_g = 10 * loss_l1 + loss_p  + l_gan1
                err_g.backward()
                optimizer_G.step()
                if idx % 200 == 0:
                    print('Epoch: {}, Iter: {}, gloss: {:.5f}, l1: {:.5f}, '
                          'lp: {:.5f}, lg: {:.5f}, ld: {:.5f}'.format(epoch, idx, err_g.item(),
                                                                      loss_l1.item(), loss_p.item(),
                                                                      l_gan1.item(), err_d.item()))

                if idx % 200 == 0 or (epoch == self.epochs - 1 and idx == len(self.dataloader - 1)):
                    torch.save(self.dnet.state_dict(), './model/d_net.pth')
                    torch.save(self.model.state_dict(), './model/g_net.pth')
                    with torch.no_grad():
                        fake_ = self.model(test_img, test_pose).detach().cpu()
                        source = (test_batch['input']['t_map']).detach().cpu()
                        poses = (test_batch['input']['t_pose']).detach().cpu()
                        poses = torch.unsqueeze(torch.sum(poses, dim=1), dim=1)
                    sample_source = vutils.make_grid(source, padding=2, normalize=False)
                    sample_pose = vutils.make_grid(poses, padding=2, normalize=False)
                    sample = vutils.make_grid(fake_, padding=2, normalize=False)
                    plt.figure(figsize=(32, 16))
                    plt.axis('off')
                    plt.title('fake image')
                    plt.subplot(3, 1, 1)
                    plt.imshow(np.transpose(sample_source, (1, 2, 0)))
                    plt.subplot(3, 1, 2)
                    plt.imshow(np.transpose(sample_pose, (1, 2, 0)))
                    plt.subplot(3, 1, 3)
                    plt.imshow(np.transpose(sample, (1, 2, 0)))
                    plt.savefig("./sample/epoch_{}_iter_{}.png".format(epoch, idx))
                    plt.close()

