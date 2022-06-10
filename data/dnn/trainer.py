import os
import tqdm
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import data.dnn.utility as util

class Trainer():
    def __init__(self, args, model, trainloader, testloader, criterion, optimizer, scheduler, index=None):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.testloader = testloader
        self.writer = SummaryWriter(args.log_dir)
        if index is None:
            self.device = torch.device('cpu' if self.args.use_cpu else 'cuda')
        else:
            self.device = torch.device('cuda:{}'.format(index))

        self.visualize_graph()

    def visualize_graph(self):
        dataiter = iter(self.trainloader)
        lr, hr = dataiter.next()
        self.writer.add_graph(self.model, lr)

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.args.num_epochs):
            running_loss = 0.0
            for i, data in tqdm.tqdm(enumerate(self.trainloader), total=len(self.trainloader), desc=f'Epoch(training): {epoch+1}/{self.args.num_epochs}'):
                lr, hr = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                sr = self.model(lr)
                loss = self.criterion(sr, hr)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            self.writer.add_scalar('training loss',
                                   running_loss / 1000,
                                   (epoch + 1) * len(self.trainloader))
            self.writer.add_scalar('learning rate',
                                   *self.scheduler.get_lr(),
                                   (epoch + 1) * len(self.trainloader))
            self.validate(epoch)
            self.scheduler.step()
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict()},
                       os.path.join(self.args.checkpoint_dir, f'{self.model.name}.pt'))

    def validate(self, epoch):
        scale = self.args.scale
        with torch.no_grad():
            sr_psnr = 0
            bicubic_psnr = 0
            for i, data in tqdm.tqdm(enumerate(self.testloader), total=len(self.testloader), desc=f'Epoch(validation): {epoch+1}/{self.args.num_epochs}'):
                lr, hr = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                sr = self.model(lr)
                bicubic = F.interpolate(lr, [hr.size()[2], hr.size()[3]], mode='bicubic')

                sr_psnr += util.calc_psnr(sr, hr)
                bicubic_psnr += util.calc_psnr(bicubic, hr)

            self.writer.add_scalar('Absolute PSNR',
                                   sr_psnr / len(self.testloader),
                                   (epoch + 1) * len(self.trainloader))
            self.writer.add_scalar('PSNR Gain',
                                   (sr_psnr - bicubic_psnr) / len(self.testloader),
                                   (epoch + 1) * len(self.trainloader))

class EngorgioTrainer():
    def __init__(self, args, model, trainloader, testloader, criterion, optimizer, scheduler, index=None):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.testloader = testloader
        self.writer = SummaryWriter(args.log_dir)
        if index is None:
            self.device = torch.device('cpu' if self.args.use_cpu else 'cuda')
        else:
            self.device = torch.device('cuda:{}'.format(index))

        self.visualize_graph()

    def visualize_graph(self):
        dataiter = iter(self.trainloader)
        lr, hr = dataiter.next()
        self.writer.add_graph(self.model, lr)

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.args.num_epochs):
            running_loss = 0.0
            for i, data in tqdm.tqdm(enumerate(self.trainloader), total=len(self.trainloader), desc=f'Epoch(training): {epoch+1}/{self.args.num_epochs}'):
                lr, hr = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                sr = self.model(lr)
                loss = self.criterion(sr, hr)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            self.writer.add_scalar('training loss',
                                   running_loss / 1000,
                                   (epoch + 1) * len(self.trainloader))
            self.writer.add_scalar('learning rate',
                                   *self.scheduler.get_lr(),
                                   (epoch + 1) * len(self.trainloader))
            self.validate(epoch)
            self.scheduler.step()
            if (epoch + 1) % 5 == 0:
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict()},
                        os.path.join(self.args.checkpoint_dir, f'{self.model.name}_e{epoch+1}.pt'))

    def validate(self, epoch):
        scale = self.args.scale
        with torch.no_grad():
            sr_psnr = 0
            bicubic_psnr = 0
            for i, data in tqdm.tqdm(enumerate(self.testloader), total=len(self.testloader), desc=f'Epoch(validation): {epoch+1}/{self.args.num_epochs}'):
                lr, hr = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                sr = self.model(lr)
                bicubic = F.interpolate(lr, [hr.size()[2], hr.size()[3]], mode='bicubic')

                sr_psnr += util.calc_psnr(sr, hr)
                bicubic_psnr += util.calc_psnr(bicubic, hr)

            self.writer.add_scalar('Absolute PSNR',
                                   sr_psnr / len(self.testloader),
                                   (epoch + 1) * len(self.trainloader))
            self.writer.add_scalar('PSNR Gain',
                                   (sr_psnr - bicubic_psnr) / len(self.testloader),
                                   (epoch + 1) * len(self.trainloader))

