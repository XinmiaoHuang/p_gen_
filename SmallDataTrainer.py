import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from base.base_trainer import BaseTrainer
from loss.dloss import CriterionD
from loss.ganloss import GANLoss
import random
import os, cv2


class SmallDataTrainer(BaseTrainer):
    def __init__(self, opt, dataloader, criterion, model):
        super().__init__(opt, dataloader, criterion, model.gnet)
        self.dnet = model.dnet
        self.dpnet = model.dpnet
        self.len = 1200
        self.criterion_d = CriterionD()
        self.criterion_dp = CriterionD()
        self.criterionGAN = GANLoss(use_lsgan=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_dataset()

    def load_dataset(self):
        images = []
        tposes = []
        targets = []
        for i in range(self.len):
            img_dir = os.path.join('./small_data/imgs/', str(i) + '.jpg')
            image = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_shape[:2])
            image = image * 1. / 255
            image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)
            pose_dir = os.path.join('./small_data/poses/', str(i) + '.jpg')
            t_pose = cv2.cvtColor(cv2.imread(pose_dir), cv2.COLOR_BGR2RGB)
            t_pose = cv2.resize(t_pose, self.img_shape[:2])
            t_pose = t_pose * 1. / 255
            t_pose = np.expand_dims(t_pose.transpose((2, 0, 1)), axis=0)
            target_dir = os.path.join('./small_data/target/', str(i) + '.jpg')
            target = cv2.cvtColor(cv2.imread(target_dir), cv2.COLOR_BGR2RGB)
            target = cv2.resize(target, self.img_shape[:2])
            target = target * 1. / 255
            target = np.expand_dims(target.transpose((2, 0, 1)), axis=0)
            images.append(image)
            tposes.append(t_pose)
            targets.append(target)
        self.images = np.array(images)
        self.tposes = np.array(tposes)
        self.targets = np.array(targets)

    def get_batch(self):
        rnd_n = random.randint(0, self.len-1)
        image = self.images[rnd_n]
        tpose = self.tposes[rnd_n]
        target = self.targets[rnd_n]
        bsize = 16
        for i in range(bsize-1):
            rnd_n = random.randint(0, self.len - 1)
            image = np.vstack((image, self.images[rnd_n]))
            tpose = np.vstack((tpose, self.tposes[rnd_n]))
            target = np.vstack((target, self.targets[rnd_n]))
        image = torch.from_numpy(image)
        tpose = torch.from_numpy(tpose)
        target = torch.from_numpy(target)
        return image, tpose, target

    def train(self):
        print(f'Use device: {self.device}'
              f'Network:\n'
              f'In channels: {self.n_channels}  '
              f'Out channels: {self.n_classes}')
        self.model.to(device=self.device)
        self.dnet.to(device=self.device)
        self.dpnet.to(device=self.device)

        self.optimizer_G = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_D = torch.optim.Adam(self.dnet.parameters(), lr=self.learning_rate)
        self.optimizer_DP = torch.optim.Adam(self.dpnet.parameters(), lr=self.learning_rate)

        test_batch = iter(self.dataloader).next()
        test_img = test_batch['input']['image']
        test_img = test_img.type(torch.FloatTensor).to(self.device)
        test_pose = test_batch['input']['t_pose']
        test_pose = test_pose.type(torch.FloatTensor).to(self.device)
        for epoch in range(self.epochs):
            for idx in range(10000):
                img, t_pose, target = self.get_batch()
                img = img.type(torch.FloatTensor).to(self.device)
                t_pose = t_pose.type(torch.FloatTensor).to(self.device)
                target = target.type(torch.FloatTensor).to(self.device)

                self.dnet.zero_grad()
                real_out_d = self.dnet(target)
                label_d = torch.full(real_out_d.size(), 1, device=self.device)
                real_loss_d = self.criterion_d(real_out_d, label_d)
                real_loss_d.backward()

                fake = self.model(img, t_pose)
                fake_out_d = self.dnet(fake.detach())
                label_d.fill_(0)
                fake_loss_d = self.criterion_d(fake_out_d, label_d)
                fake_loss_d.backward()

                err_d = real_loss_d + fake_loss_d
                self.optimizer_D.step()
                # err_dp, label_dp = self.optimize_d2(target, fake)

                self.model.zero_grad()
                fake_score_d = self.dnet(fake)
                # fake_score_dp = self.dpnet(fake)
                label_d.fill_(1)
                # label_dp.fill_(1)
                l_gan1 = self.criterionGAN(fake_score_d, label_d)
                # l_gan2 = self.criterionGAN(fake_score_dp, label_dp)
                loss_l1, loss_p = self.criterionPerPixel(fake, target)
                # err_g = 0.5 * l_gan1 + 0.5 * l_gan2 + 25 * loss_l1
                err_g = 40 * loss_l1 + loss_p + l_gan1
                err_g.backward()
                self.optimizer_G.step()
                if idx % 200 == 0:
                    # print('Iter: {}, gloss: {:.5f}, lgan1: {:.5f}, lgan2: {:.5f}, loss_l1: {:.5f}, '
                    #       'dloss: {:.5f}, dloss_p:{:.5f}'.format(idx, err_g.item(), l_gan1.item(), l_gan2.item(),
                    #                                              loss_l1.item(), err_d.item(), err_dp.item()))
                    # print('Iter: {}, gloss: {:.5f}, l1: {:.5f}, '
                    #       'lp: {:.5f}'.format(idx, err_g.item(),
                    #                           loss_l1.item(), loss_p.item()))
                    print('Iter: {}, gloss: {:.5f}, l1: {:.5f}, '
                          'lp: {:.5f}, lg: {:.5f}, ld: {:.5f}'.format(idx, err_g.item(),
                                              loss_l1.item(), loss_p.item(),
                                              l_gan1.item(), err_d.item()))
                if idx % 200 == 0:
                    torch.save(self.dnet.state_dict(), './model/d_net.pth')
                    torch.save(self.model.state_dict(), './model/g_net.pth')
                    # torch.save(self.dpnet.state_dict(), './model/dp_net.pth')
                    with torch.no_grad():
                        source = (test_batch['input']['target']).detach().cpu()
                        fake_ = self.model(test_img, test_pose).detach().cpu()
                        poses = (test_batch['input']['t_pose']).detach().cpu()
                    sample_source = vutils.make_grid(source, padding=2, normalize=False)
                    sample = vutils.make_grid(fake_, padding=2, normalize=False)
                    sample_pose = vutils.make_grid(poses, padding=2, normalize=False)
                    plt.figure(figsize=(64, 64))
                    # plt.ion()
                    plt.axis('off')
                    plt.title('fake image')
                    plt.subplot(3, 1, 1)
                    plt.imshow(np.transpose(sample_source, (1, 2, 0)))
                    plt.subplot(3, 1, 2)
                    plt.imshow(np.transpose(sample, (1, 2, 0)))
                    plt.subplot(3, 1, 3)
                    plt.imshow(np.transpose(sample_pose, (1, 2, 0)))
                    plt.savefig("./sample/iter_{}.png".format(idx))
                    # plt.pause(2)
                    # plt.close()

    def optimize_d2(self, real, fake):
        self.dpnet.zero_grad()
        real_out_d = self.dpnet(real)
        label_d = torch.full(real_out_d.size(), 1, device=self.device)
        real_loss_d = self.criterion_dp(real_out_d, label_d)
        real_loss_d.backward()

        fake_out_d = self.dpnet(fake.detach())
        label_d.fill_(0)
        fake_loss_d = self.criterion_dp(fake_out_d, label_d)
        fake_loss_d.backward()

        err_d2 = real_loss_d + fake_loss_d
        self.optimizer_DP.step()
        return err_d2, label_d
