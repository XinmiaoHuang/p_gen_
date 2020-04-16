import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from base.base_trainer import BaseTrainer
from loss.dloss import CriterionD
from loss.ganloss import GANLoss
from numpy import mean


class Stage2Trainer(BaseTrainer):
    def __init__(self, opt, dataloader, criterion, model):
        super().__init__(opt, dataloader, criterion, model.gnet)
        self.dnet = model.dnet
        self.criterion_d = CriterionD()
        self.criterionGAN = GANLoss(use_lsgan=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        print(f'Use device: {self.device}'
              f'Network:\n'
              f'In channels: {self.n_channels}  '
              f'Out channels: {self.n_classes}')
        self.model.to(device=self.device)
        self.dnet.to(device=self.device)

        optimizer_G = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer_D = torch.optim.Adam(self.dnet.parameters(), lr=self.learning_rate)

        test_batch = iter(self.dataloader).next()
        test_img = test_batch['input']['image']
        test_img = test_img.type(torch.FloatTensor).to(self.device)
        test_pose = test_batch['target']['pose']
        test_pose = test_pose.type(torch.FloatTensor).to(self.device)
        test_map = test_batch['target']['t_map']
        test_map = test_map.type(torch.FloatTensor).to(self.device)
        D_loss = []
        G_loss = []
        for epoch in range(self.epochs):
            for idx, data_batch in enumerate(self.dataloader):
                img = data_batch['input']['image']
                t_pose = data_batch['target']['pose']
                t_map = data_batch['target']['t_map']
                target = data_batch['target']['image']
                img = img.type(torch.FloatTensor).to(self.device)
                t_pose = t_pose.type(torch.FloatTensor).to(self.device)
                t_map = t_map.type(torch.FloatTensor).to(self.device)
                target = target.type(torch.FloatTensor).to(self.device)

                self.dnet.zero_grad()
                real_out_d = self.dnet(target)
                label_d = torch.full(real_out_d.size(), 1, device=self.device)
                real_loss_d = self.criterion_d(real_out_d, label_d)
                real_loss_d.backward()

                fake = self.model(img, t_pose, t_map)
                fake_out_d = self.dnet(fake.detach())
                label_d.fill_(0)
                fake_loss_d = self.criterion_d(fake_out_d, label_d)
                fake_loss_d.backward()

                err_d = real_loss_d + fake_loss_d
                D_loss.append(err_d.item())
                optimizer_D.step()

                self.model.zero_grad()
                fake_score_d = self.dnet(fake)
                label_d.fill_(1)
                l_gan1 = self.criterionGAN(fake_score_d, label_d)
                loss_l1, loss_p = self.criterionPerPixel(fake, target)
                err_g = 10 * loss_l1 + loss_p
                G_loss.append(err_g.item())
                err_g.backward()
                optimizer_G.step()
                if idx % 200 == 0:
                    print('Iter: {}, gloss: {:.5f}, l1: {:.5f}, '
                          'lp: {:.5f}, lg: {:.5f}, ld: {:.5f}'.format(idx, mean(G_loss), mean(D_loss),
                                                                      loss_l1.item(), loss_p.item(),
                                                                      l_gan1.item()))
                    D_loss.clear()
                    G_loss.clear()

                if idx % 200 == 0 or (epoch == self.epochs - 1 and idx == len(self.dataloader - 1)):
                    torch.save(self.dnet.state_dict(), './model/d_net.pth')
                    torch.save(self.model.state_dict(), './model/g_net.pth')
                    with torch.no_grad():
                        source = (test_batch['target']['image']).detach().cpu()
                        fake_ = self.model(test_img, test_pose, test_map).detach().cpu()
                        poses = (test_batch['target']['pose']).detach().cpu()
                        poses = torch.unsqueeze(torch.sum(poses, dim=1), dim=1)
                    sample_source = vutils.make_grid(source, padding=2, normalize=False)
                    sample = vutils.make_grid(fake_, padding=2, normalize=False)
                    sample_pose = vutils.make_grid(poses, padding=2, normalize=False)
                    plt.figure(figsize=(64, 64))
                    plt.axis('off')
                    plt.title('fake image')
                    plt.subplot(3, 1, 1)
                    plt.imshow(np.transpose(sample_source, (1, 2, 0)))
                    plt.subplot(3, 1, 2)
                    plt.imshow(np.transpose(sample, (1, 2, 0)))
                    plt.subplot(3, 1, 3)
                    plt.imshow(np.transpose(sample_pose, (1, 2, 0)))
                    plt.savefig("./sample/iter_{}.png".format(idx))
                    plt.close()

