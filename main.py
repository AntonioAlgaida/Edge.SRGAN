# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:52:34 2021

@author: Antonio Guillen-Perez
@twitter: agnprz
@email: antonio_algaida@hotmail.com
"""

from torch import nn, optim
from tqdm import tqdm
import torch.nn.functional as F
import data_loader as data_loader
import torchvision
from pytorch_msssim import SSIM
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision.utils import save_image
import utils
import torch


# Edge Generator
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class EdgeGenerator(nn.Module):
    def __init__(self, scale=4, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
            )

    def forward(self, lr_images, lr_edges):
        hr_images = F.interpolate(lr_images, scale_factor=4)
        hr_edges = F.interpolate(lr_edges, scale_factor=4)
        x = torch.cat((hr_images, hr_edges), dim=1)

        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


def sigmoid_mul(x):
    return x * F.sigmoid(x)


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            *list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x):
        y = self.prelu(self.bn1(self.conv1(x)))

        return self.bn2(self.conv2(y)) + x


class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x):
        return self.prelu(self.shuffler(self.conv(x)))


# HR Generator
class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor=4):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(int(self.upsample_factor/2)):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(65, 128, 3, stride=1, padding=1)

        for i in range(self.n_residual_blocks//4):
            self.add_module('residual_block_edges' + str(i+1), residualBlock(in_channels=128, n=128))

        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 3, 9, stride=1, padding=4)

        self.prelu = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, lr_img, hr_edges):
        x = self.prelu(self.conv1(lr_img))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(self.upsample_factor/2)):
            x = self.__getattr__('upsample' + str(i+1))(x)

        x = torch.cat((x, hr_edges), dim=1)
        x = self.prelu(self.conv3(x))

        y = x.clone()
        for i in range(self.n_residual_blocks//4):
            y = self.__getattr__('residual_block_edges' + str(i+1))(y)

        x = self.bn4(self.conv4(y)) + x

        return torch.tanh(self.conv5(x))


# HR Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = sigmoid_mul(self.conv1(x))
        x = sigmoid_mul(self.bn2(self.conv2(x)))
        x = sigmoid_mul(self.bn3(self.conv3(x)))
        x = sigmoid_mul(self.bn4(self.conv4(x)))
        x = sigmoid_mul(self.bn5(self.conv5(x)))
        x = sigmoid_mul(self.bn6(self.conv6(x)))
        x = sigmoid_mul(self.bn7(self.conv7(x)))
        x = sigmoid_mul(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


def cuda(*args):
    return (item.cuda() for item in args)


def train():
    writer = SummaryWriter()

    BATCH_SIZE = 8
    dataloaders = data_loader.fetch_dataloader('train', batch_size=BATCH_SIZE, nworks=2, max_patch=30)
    train_dl = dataloaders['train']

    CUDA = torch.cuda.is_available()
    n_gen = 20
    generator = Generator(n_gen)

    discriminator = Discriminator()
    
    edge_path = 'E:\\Hackathon\\ckpts\\EdgeModel_gen.pth'
    edge_generator = EdgeGenerator(use_spectral_norm=True).cuda()
    data = torch.load(edge_path)
    edge_generator.load_state_dict(data['generator'])
    edge_generator.eval()
    
    # For the content loss
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
    # content_criterion = pytorch_ssim.SSIM()
    pre_train_content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()
    feature_criterion = nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()
    ones_const = Variable(torch.ones(BATCH_SIZE, 1))

    # if gpu is to be used
    if CUDA:
        generator.cuda()
        discriminator.cuda()
        feature_extractor.cuda()
        criterion_pixel.cuda()
        # content_criterion.cuda()
        pre_train_content_criterion.cuda()
        feature_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=1e-4)  # , lr=1e-4)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4)  # , lr=1e-4)

    # set model to training mode
    generator.train()
    discriminator.train()

    # summary for current training loop and a running average object for loss
    pre_g_loss_avg = utils.RunningAverage()
    g_loss_avg = utils.RunningAverage()
    d_loss_avg = utils.RunningAverage()
    mse_loss_avg = utils.RunningAverage()
    ssim_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    PRE_EPOCHS = 2  # 1 or 2 epochs is more than enough
    EPOCHS = 100
    best = 1
    model_dir = 'ckpts/'
    i = 0

    # Pre-train generator using raw MSE loss to speed up the training
    pretrain = True
    if pretrain:
        print('Generator pre-training')
        with tqdm(total=PRE_EPOCHS*len(train_dl), smoothing=0.1) as t:
            for epoch in range(PRE_EPOCHS):
                for items in train_dl:
                    lr_images, hr_images, lr_edges, hr_edges = cuda(*items)
                    i += 1
                    # Generate real and fake inputs
                    high_res_real = Variable(hr_images)
                    hr_edges = edge_generator(lr_images, lr_edges).detach()
                    high_res_fake = generator(Variable(lr_images), Variable(hr_edges*2-1))

                    ######### Train generator #########
                    generator.zero_grad()
                    generator_content_loss = pre_train_content_criterion(
                                                high_res_fake, high_res_real)
                    generator_content_loss.backward()
                    optim_generator.step()

                    pre_g_loss_avg.update(generator_content_loss.item())

                    ######### Status and display #########
                    t.set_postfix(EPOCH='{}'.format(epoch),
                                  gen_loss='{:05.6f}'.format(pre_g_loss_avg()))
                    writer.add_scalar('Loss/generator_content_loss', generator_content_loss.item(), i)
                    t.update()

                    if (i+1) % 100 == 0:
                        # Save model checkpoints
                        flag = f'pre_G_{n_gen}_{epoch}'
                        utils.save_checkpoint({'epoch': epoch + 1,
                                               'state_dict': generator.state_dict(),
                                               'optim_dict': optim_generator.state_dict()},
                                              checkpoint=model_dir, is_best=False, flag=flag)

                    if (i+1) % 100 == 0:
                        # Save image grid with upsampled inputs and ESRGAN outputs
                        imgs_lr = nn.functional.interpolate(lr_images, scale_factor=4)
                        # torch.cat((imgs_lr, gen_hr), -1)
                        img_grid = torch.cat((((imgs_lr+1)/2).cuda(), (high_res_fake+1)/2, (high_res_real+1)/2), -1)
                        save_image(img_grid, f"imgs/pretraining/{i}.png", nrow=1, normalize=False)

    BATCH_SIZE = 8
    dataloaders = data_loader.fetch_dataloader('train', batch_size=BATCH_SIZE, nworks=2)
    train_dl = dataloaders['train']

    load_pretrain = False
    if load_pretrain:
        PATH = f'ckpts/pre_G_20_0_last.pth.tar'

        checkpoint = torch.load(PATH)
        generator.load_state_dict(checkpoint['state_dict'])
        optim_generator.load_state_dict(checkpoint['optim_dict'])

    load = False
    if load:
        PATH = 'ckpts/G_edges2_20_21_last.pth.tar'
        # PATH = f'ckpts/G_16_best_0.1326_last.pth.tar'

        checkpoint = torch.load(PATH)
        # generator = checkpoint['model']
        generator.load_state_dict(checkpoint['state_dict'])
        optim_generator.load_state_dict(checkpoint['optim_dict'])

        PATH = 'ckpts/D_edges2_20_21_last.pth.tar'
        # PATH = f'ckpts/D_16_best_0.1326_last.pth.tar'
        checkpoint = torch.load(PATH)
        # discriminator = checkpoint['model']
        discriminator.load_state_dict(checkpoint['state_dict'])
        optim_discriminator.load_state_dict(checkpoint['optim_dict'])
        
        epoch = checkpoint['epoch']
        i = epoch*len(train_dl)
        LR = 1e-4
        for param_group in optim_generator.param_groups:
            param_group['lr'] = LR
        for param_group in optim_discriminator.param_groups:
            param_group['lr'] = LR
    
    else:
        i=1
        LR = 1e-4

    # Main train
    accumulation_steps = 16//BATCH_SIZE
    print('Main training')
    ssim_module = SSIM(data_range=2, size_average=True, channel=3).cuda()
    with tqdm(total=EPOCHS*len(train_dl), smoothing=0.1) as t:
        for epoch in range(EPOCHS):
            # for train_batch, labels_batch in train_dl:
            for items in train_dl:
                lr_images, hr_images, lr_edges, hr_edges = cuda(*items)
                i += 1

                # I will reduce the LR each 250.000 steps
                if i % 250e3 == 0:
                    LR = LR/2
                    for param_group in optim_generator.param_groups:
                        param_group['lr'] = LR
                    for param_group in optim_discriminator.param_groups:
                        param_group['lr'] = LR
                    print(f'Reducing LR: {LR}')

                try:
                    high_res_real = Variable(hr_images)
                    
                    hr_edges_pred = edge_generator(lr_images, lr_edges).detach()
                    
                    high_res_fake = generator(Variable(lr_images), Variable(hr_edges_pred*2-1))
                    
                    BATCH_SIZE = lr_images.shape[0]
                    target_real = Variable(torch.rand(BATCH_SIZE, 1)*0.5 + 0.7).cuda()
                    target_fake = Variable(torch.rand(BATCH_SIZE, 1)*0.3).cuda()

                    ######### Train discriminator #########
                    discriminator.zero_grad()

                    discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                                        adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)

                    discriminator_loss.backward()

                    if (i+1) % accumulation_steps == 0:
                        optim_discriminator.step()
                        optim_discriminator.zero_grad()

                    
                    ######### Train generator #########
                    generator.zero_grad()
                    real_features = Variable(feature_extractor(data_loader.normalize(((high_res_real+1)/2).data)))
                    fake_features = feature_extractor(data_loader.normalize(((high_res_fake+1)/2).data))

                    ssim_loss = 1 - ssim_module(high_res_fake, high_res_real) #- to maximize ssim, +1 to have between positive values
                    generator_content_loss = criterion_pixel(high_res_fake, high_res_real) + 0.006*feature_criterion(fake_features, real_features) + ssim_loss

                    try:
                        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
                    except:
                        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake),
                                                                           Variable(torch.ones(lr_images.shape[0], 1)).cuda())

                    # loss_pixel = criterion_pixel(high_res_fake, high_res_real)

                    generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss #+ 1e-2*loss_pixel
                    generator_total_loss.backward()
                    # optim_generator.step()

                    if (i+1) % accumulation_steps == 0:
                        optim_generator.step()
                        optim_generator.zero_grad()


                    g_loss_avg.update(generator_total_loss.item())
                    d_loss_avg.update(discriminator_loss.item())
                    mse_loss_avg.update(generator_content_loss.item())
                    ssim_avg.update(ssim_loss)
                    current = ssim_avg()
                    t.set_postfix(gen_loss='{:05.6f}'.format(current), dis_loss='{:05.4f}'.format(
                        d_loss_avg()), best='{:05.4f}'.format(best))

                    if (i+1) % 10 == 0:
                        writer.add_scalar('loss/BEST', best, i)
                        writer.add_scalar('loss/SSIM', ssim_avg(), i)
                        writer.add_scalar('loss/generator_total_loss', g_loss_avg(), i)
                        writer.add_scalar('loss/discriminator_loss',  d_loss_avg(), i)
                        writer.add_scalar('loss/generator_content_loss', mse_loss_avg(), i)
                    t.update()
                except FileNotFoundError:
                    print("Wrong file or file path")
                else:
                    pass

                if (i+1) % 500 == 0:
                    # Save model checkpoints each 500 epochs

                    flag = f'G_edges2_{n_gen}_{epoch}'
                    utils.save_checkpoint({'epoch': epoch + 1,
                                           'state_dict': generator.state_dict(),
                                           'optim_dict': optim_generator.state_dict()},
                                          checkpoint=model_dir, is_best=False, flag=flag)
                    flag = f'D_edges2_{n_gen}_{epoch}'
                    utils.save_checkpoint({'epoch': epoch + 1,
                                           'state_dict': discriminator.state_dict(),
                                           'optim_dict': optim_discriminator.state_dict()},
                                          checkpoint=model_dir, is_best=False, flag=flag)

                if (i+1) % 100 == 0:
                    # Save image grid with upsampled inputs and SRGAN outputs
                    imgs_lr = nn.functional.interpolate(lr_images, scale_factor=4)
                    img_grid = torch.cat((((imgs_lr+1)/2).cuda(), (high_res_fake+1)/2, (high_res_real+1)/2), -1)
                    save_image(img_grid, f"imgs/training/{i}.png", nrow=1, normalize=False)
                    
                    edges_lr = nn.functional.interpolate(lr_edges, scale_factor=4)
                    # torch.cat((imgs_lr, gen_hr), -1)
                    img_grid = torch.cat((edges_lr.cuda(), hr_edges_pred, hr_edges), -1)
                    save_image(img_grid, f"imgs/training/{i}_edges.png", nrow=1, normalize=False)

                if current < best and i > 1000 and (i+1) % 100 == 0:
                    flag = f'G_{n_gen}_best2_{current:05.4f}'
                    utils.save_checkpoint({'epoch': epoch + 1,
                                           'state_dict': generator.state_dict(),
                                           'optim_dict': optim_generator.state_dict()},
                                          checkpoint=model_dir, is_best=False, flag=flag)
                    flag = f'D_{n_gen}_best2_{current:05.4f}'
                    utils.save_checkpoint({'epoch': epoch + 1,
                                           'state_dict': discriminator.state_dict(),
                                           'optim_dict': optim_discriminator.state_dict()},
                                          checkpoint=model_dir, is_best=False, flag=flag)
                    best = current
# %


def test():
    CUDA = torch.cuda.is_available()
    n_gen = 20

    generator = Generator(n_gen)

    if CUDA:
        generator.cuda()

    PATH = f'ckpts/G_{n_gen}_best_0.1521_last.pth.tar'

    checkpoint = torch.load(PATH)
    generator.load_state_dict(checkpoint['state_dict'])
    generator.eval()

    edge_path = 'E:/Hackathon/ckpts/EdgeModel_gen.pth'
    edge_generator = EdgeGenerator(use_spectral_norm=True).cuda()
    data = torch.load(edge_path)
    edge_generator.load_state_dict(data['generator'])
    edge_generator.eval()
    
    dataloaders = data_loader.fetch_dataloader('test', nworks=2)
    test_dl = dataloaders['test']

    with tqdm(total=len(test_dl)) as t:
        for lr_images, lr_edges, fname in test_dl:
            lr_images, lr_edges = cuda(*[lr_images, lr_edges])
            # compute model output
            hr_edges_pred = edge_generator(lr_images, lr_edges).detach()
            high_res_fake = generator(Variable(lr_images), Variable(hr_edges_pred*2-1))
            fname = fname[0].replace('croppedoverl', 'output')

            pad = 8*2
            save_image((high_res_fake[0][:, pad:-pad, pad:-pad]+1)/2, fname)
            t.update()


# %%
if __name__ == '__main__':
    import traceback

    try:
        train()
        #test()
    except Exception as e:
        print(e)
        print(traceback.print_exc())
