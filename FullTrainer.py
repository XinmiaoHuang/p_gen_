import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from base.base_trainer import BaseTrainer
from loss.dloss import CriterionD
from loss.ganloss import GANLoss
from base.utils import make_parts_shape


class FullTrainer(BaseTrainer):
    def __init__(self, opt, dataloader, criterion, model, testloader):
        super().__init__(opt, dataloader, criterion, model.gnet)
        self.testloader = testloader
        self.dnet = model.dnet
        self.dnet_s = model.dnet_s
        self.criterion_d = CriterionD(use_lsgan=True)
        self.criterionGAN = GANLoss(use_lsgan=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        print(f'Use device: {self.device}'
              f'Network:\n'
              f'In channels: {self.n_channels}  '
              f'Out channels: {self.n_classes}')
        self.model.to(device=self.device)
        self.dnet.to(device=self.device)
        self.dnet_s.to(device=self.device)

        optimizer_G = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer_D = torch.optim.Adam(self.dnet.parameters(), lr=self.learning_rate)
        optimizer_Ds = torch.optim.Adam(self.dnet_s.parameters(), lr=self.learning_rate)

        test_batch = iter(self.testloader).next()
        test_img = test_batch['input']['image']
        test_img = test_img.type(torch.FloatTensor).to(self.device)
        test_pose = test_batch['input']['t_pose']
        test_pose = test_pose.type(torch.FloatTensor).to(self.device)
        test_map = test_batch['input']['s_map']
        test_map = test_map.type(torch.FloatTensor).to(self.device)
        for epoch in range(self.epochs):
            for idx, data_batch in enumerate(self.dataloader):
                img = data_batch['input']['image']
                smap = data_batch['input']['s_map']
                t_pose = data_batch['input']['t_pose']
                target = data_batch['input']['target']
                t_map = data_batch['input']['t_map']
                img = img.type(torch.FloatTensor).to(self.device)
                t_pose = t_pose.type(torch.FloatTensor).to(self.device)
                target = target.type(torch.FloatTensor).to(self.device)
                smap = smap.type(torch.FloatTensor).to(self.device)
                t_map = t_map.type(torch.FloatTensor).to(self.device)
                # image discriminator
                self.dnet.zero_grad()
                real_out_d = self.dnet(target)
                label_d = torch.full(real_out_d.size(), 1, device=self.device)
                real_loss_d = self.criterion_d(real_out_d, label_d)
                real_loss_d.backward()

                fake, fake_smap = self.model(img, t_pose, smap)
                fake_out_d = self.dnet(fake.detach())
                label_d.fill_(0)
                fake_loss_d = self.criterion_d(fake_out_d, label_d)
                fake_loss_d.backward()

                err_d = real_loss_d + fake_loss_d
                optimizer_D.step()

                # semantic map discriminator
                self.dnet_s.zero_grad()
                real_out_d = self.dnet_s(t_map)
                label_d = torch.full(real_out_d.size(), 1, device=self.device)
                real_loss_d = self.criterion_d(real_out_d, label_d)
                real_loss_d.backward()

                fake_out_d = self.dnet_s(fake_smap.detach())
                label_d.fill_(0)
                fake_loss_d = self.criterion_d(fake_out_d, label_d)
                fake_loss_d.backward()

                err_ds = real_loss_d + fake_loss_d
                optimizer_Ds.step()

                self.model.zero_grad()
                fake_score_d = self.dnet(fake)
                fake_score_ds = self.dnet_s(fake_smap)
                label_d.fill_(1)
                l_gan1 = self.criterionGAN(fake_score_d, label_d)
                l_gan2 = self.criterionGAN(fake_score_ds, label_d)
                loss_l1, loss_p = self.criterionPerPixel(fake, target)
                loss_l1_s, loss_ps = self.criterionPerPixel(fake_smap, t_map)
                err_g = 10 * loss_l1 + 0.8 * loss_p # + 0.1 * l_gan1 + l_gan2
                err_g.backward()
                optimizer_G.step()
                if idx % 200 == 0:
                    print('Epoch: {}, Iter: {}, gloss: {:.5f}, l1: {:.5f}, l1_s: {:.5f},'
                          'lp: {:.5f}, lps: {:.5f}, lg: {:.5f}, lg2: {:.5f}, ld: {:.5f}, lds: {:.5f}'.format(
                           epoch, idx, err_g.item(), loss_l1.item(), loss_l1_s.item(), loss_p.item(), loss_ps.item(),
                           l_gan1.item(), l_gan2.item(), err_d.item(), err_ds.item()))

                if idx % 200 == 0 or (epoch == self.epochs - 1 and idx == len(self.dataloader - 1)):
                    torch.save(self.dnet.state_dict(), './model/d_net.pth')
                    torch.save(self.model.state_dict(), './model/g_net.pth')
                    with torch.no_grad():
                        fake_, fake_map = self.model(test_img, test_pose, test_map)
                        fake_ = fake_.detach().cpu()
                        fake_map = fake_map.detach().cpu()
                        target_ = (test_batch['input']['target']).detach().cpu()
                        source = (test_batch['input']['t_map']).detach().cpu()
                        poses = (test_batch['input']['t_pose']).detach().cpu()
                        poses = torch.unsqueeze(torch.sum(poses, dim=1), dim=1)
                    sample_target = vutils.make_grid(target_, padding=2, normalize=False)
                    sample_source = vutils.make_grid(source, padding=2, normalize=False)
                    sample_pose = vutils.make_grid(poses, padding=2, normalize=False)
                    sample = vutils.make_grid(fake_, padding=2, normalize=False)
                    sample_map = vutils.make_grid(fake_map, padding=2, normalize=False)
                    plt.figure(figsize=(32, 16))
                    plt.axis('off')
                    plt.title('fake image')
                    plt.subplot(5, 1, 1)
                    plt.imshow(np.transpose(sample_source, (1, 2, 0)))
                    plt.subplot(5, 1, 2)
                    plt.imshow(np.transpose(sample_pose, (1, 2, 0)))
                    plt.subplot(5, 1, 3)
                    plt.imshow(np.transpose(sample, (1, 2, 0)))
                    plt.subplot(5, 1, 4)
                    plt.imshow(np.transpose(sample_target, (1, 2, 0)))
                    plt.subplot(5, 1, 5)
                    plt.imshow(np.transpose(sample_map, (1, 2, 0)))
                    plt.savefig("./sample/epoch_{}_iter_{}.png".format(epoch, idx))
                    plt.close()

