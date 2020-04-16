import torch
from torch.utils.data import DataLoader
from base.custom_parser import CustomParser
from model import Network
from Dataset import DeepfashionPoseDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_list):
        super().__init__()
        self.model = model
        self.desire_layers = layer_list

    def forward(self, img, pose, map):
        outputs = []
        layers = self.model._modules
        x1 = layers['branch_i'](img)
        # outputs.append(x1)
        x2 = layers['branch_p'](pose)
        # outputs.append(x2)
        x_s = layers['branch_s'](map)
        # outputs.append(x_s)
        x3 = layers['resnet'][0](torch.cat((x1, x2), dim=1))
        x_s = layers['resnet2'][0](torch.cat((x2, x_s), dim=1))
        attn_map = layers['bridge'][0](x_s)
        x3 = torch.mul(x3, attn_map)
        for i in range(1, len(layers['resnet'])):
            x3 = layers['resnet'][i](x3)
            x_s = layers['resnet2'][i](x_s)
            attn_map = layers['bridge'][i](x_s)
            x3 = torch.mul(x3, attn_map)
        for i in range(len(layers['decoder'])-1):
            x3 = layers['decoder'][i](x3)
            x_s = layers['decoder2'][i](x_s)
            outputs.append(x3)
            # outputs.append(x3)
        return outputs


if __name__ == '__main__':
    parser = CustomParser()
    opt = parser.parse()
    # deepfashion = dset.ImageFolder(root='D:\\Dataset\\deepfashion_test',
    #                           transform=transforms.Compose([transforms.Resize(opt.img_height),
    #                                                         transforms.CenterCrop(opt.img_height),
    #                                                         transforms.ToTensor(),
    #                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    base_dir = 'D:/Dataset/deepfashion'
    index_dir = 'D:/Dataset/deepfashion/index.p'
    map_dir = 'D:/Dataset/deepmap_test'
    deepfashion = DeepfashionPoseDataset((opt.img_height, opt.img_width, opt.n_classes), base_dir, index_dir, map_dir)
    fashionloader = DataLoader(deepfashion, batch_size=opt.train_bsize, shuffle=True, num_workers=4, drop_last=True)
    deepfashion_test = DeepfashionPoseDataset((opt.img_height, opt.img_width, opt.n_classes),
                                              base_dir, index_dir, map_dir, training=False)
    testloader = DataLoader(deepfashion_test, batch_size=opt.test_bsize, shuffle=True, num_workers=4, drop_last=True)
    unet = Network(opt)
    unet.load_model(opt.checkpoint)
    print(f'Model loaded from {opt.checkpoint}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_batch = iter(testloader).next()
    test_img = test_batch['input']['image']
    test_img = test_img.type(torch.FloatTensor).to(device)
    test_pose = test_batch['input']['t_pose']
    test_pose = test_pose.type(torch.FloatTensor).to(device)
    test_map = test_batch['input']['s_map']
    test_map = test_map.type(torch.FloatTensor).to(device)

    model = unet.gnet.to(device)

    fake_, fake_map = model(test_img, test_pose, test_map)
    fake_ = fake_.detach().cpu().numpy()
    fake_map = fake_map.detach().cpu().numpy()
    source = (test_batch['input']['t_map']).detach().cpu().numpy()
    poses = (test_batch['input']['t_pose']).detach().cpu().numpy()
    poses = np.sum(poses[0], axis=0)
    plt.figure(figsize=(32, 16))
    plt.axis('off')
    plt.title('fake image')
    plt.subplot(3, 1, 1)
    plt.imshow(np.transpose(source[0], (1, 2, 0)))
    plt.subplot(3, 1, 2)
    plt.imshow(np.transpose(fake_[0], (1, 2, 0)))
    plt.subplot(3, 1, 3)
    plt.imshow(poses)
    plt.show()
    plt.close()
    desire_o = FeatureExtractor(model, ['branch_i'])
    o = desire_o(test_img, test_pose, test_map)

    for k in range(len(o)):
        act = o[k].detach().cpu().numpy()
        img = act[0]
        fig = plt.figure(figsize=(16, 16))
        nrow = 6
        for i in range(nrow):
            for j in range(nrow):
                plt.subplot(nrow, nrow, i * nrow + j + 1)
                plt.axis('off')
                show = img[i * 4 + j]
                plt.imshow(show)
        plt.show()
        plt.close()
