import torch
from torch import optim
import os
import sys
import time


class BaseTrainer:
    def __init__(self, config, dataloader, criterion, model):
        self.initialize(config)
        self.dataloader = dataloader
        self.criterionPerPixel = criterion
        self.model = model

    def initialize(self, config):
        self.batch_size = config.train_bsize
        self.img_shape = (config.img_height, config.img_width, config.n_channels)
        self.n_channels = config.n_channels
        self.n_classes = config.n_classes
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate
        self.bilinear = config.bilinear

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device {device}')
        print(f'Network:\n'
              f'\t{self.n_channels} input channels\n'
              f'\t{self.n_classes} output channels (classes)\n'
              f'\t{"Bilinear" if self.bilinear else "Dilated conv"} upscaling')

        self.model.to(device=device)
        # faster convolutions, but more memory
        # cudnn.benchmark = True

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        try:
            for iter in range(self.epochs):
                epoch_loss = 0
                steps = 0
                iter_start_time = time.time()
                for idx, data in enumerate(self.dataloader):
                    y_pred = self.model(data)
                    loss = self.loss_fn(y_pred, data)
                    print(iter, loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    steps += 1
                iter_end_time = time.time()
                print("End of epochs {},    Time taken: {.3f},\
                    average loss: {.5f}".format(iter, iter_end_time - iter_start_time, epoch_loss / steps))
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), 'INTERRUPTED.pth')
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def test(self):
        pass
