import torch
from Dataset import DeepfashionPoseDataset
from torch.utils.data import DataLoader
from base.custom_parser import CustomParser
from GanTrainer import GanTrainer
from FullTrainer import FullTrainer
from SmallDataTrainer import SmallDataTrainer
from Stage2Trainer import Stage2Trainer
from model import Network
from loss.totalloss import CriterionPerPixel
import torchvision.datasets as dset
import torchvision.transforms as transforms
from Dataset import DeepfashionPoseDataset

if __name__ == '__main__':
    parser = CustomParser()
    opt = parser.parse()
    # deepfashion = dset.ImageFolder(root='D:\\Dataset\\deepfashion_test',
    #                           transform=transforms.Compose([transforms.Resize(opt.img_height),
    #                                                         transforms.CenterCrop(opt.img_height),
    #                                                         transforms.ToTensor(),
    #                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    
    # windows path
    # base_dir = 'D:/Dataset/deepfashion'
    # index_dir = 'D:/Dataset/deepfashion/index.p'
    # map_dir = 'D:/Dataset/deepmap_test'

    # linux path
    base_dir = '/media/homee/Data/Dataset/deepfashion'
    index_dir = '/media/homee/Data/Dataset/deepfashion/index.p'
    map_dir = '/media/homee/Data/Dataset/deepmap_test'
    deepfashion = DeepfashionPoseDataset((opt.img_height, opt.img_width, opt.n_classes), base_dir, index_dir, map_dir)
    fashionloader = DataLoader(deepfashion, batch_size=opt.train_bsize, shuffle=True, num_workers=4, drop_last=True)
    deepfashion_test = DeepfashionPoseDataset((opt.img_height, opt.img_width, opt.n_classes),
                                              base_dir, index_dir, map_dir, training=False)
    testloader = DataLoader(deepfashion_test, batch_size=opt.test_bsize, shuffle=True, num_workers=4, drop_last=True)
    unet = Network(opt)
    unet.init('xavier')

    if opt.checkpoint is not None:
        unet.load_model(opt.checkpoint)
        print(f'Model loaded from {opt.checkpoint}')
    trainer = FullTrainer(opt, fashionloader, CriterionPerPixel(), unet, testloader)
    if opt.mode == 'train':
        trainer.train()
    else:
        pass
