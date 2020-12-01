random_seed = 1
from numpy.random import seed
seed(random_seed)
import random
random.seed(random_seed)
import pandas as pd
import numpy as np 
import os 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as Vutil
from torch.autograd import Variable

reloadModels = True
nIter= 5000
imageSize = (128,128)
batchSize = 128

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
os.chdir('/content/drive/MyDrive/Pytorch DCGAN art ')
imageFolder = "./Sample/pics/" 
modelFolder = './Model/'
resultsFolder = './Results_WGAN' 
transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(imageSize),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

import h5py
 
class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, in_file, transform=None):
        super(dataset_h5, self).__init__()
 
        self.file = h5py.File(in_file, 'r')
        self.transform = transform
 
    def __getitem__(self, index):
        x = self.file['train_img'][index, ...]
        y = self.file['train_labels'][index, ...]
        
        # Preprocessing each image
        if self.transform is not None:
            x = self.transform(x)        
        
        return (x, y), index
 
    def __len__(self):
        return self.file['train_img'].shape[0]

dataset = dataset_h5("./Sample/dataset.hdf5" ,transform=transform)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchSize,drop_last=True,shuffle=True)

def weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

class Generator(nn.Module): 
    def __init__(self): 
        super(Generator, self).__init__() 
        self.iterCount = 0
        self.main = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias = False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True), 
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True), 

            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace = True), 
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.2, inplace = True), 
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(0.2, inplace = True), 
            
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias = False), 
            nn.Tanh() 
        )

    def forward(self, input): 
        output = self.main(input) 
        return output 

class Discriminator(nn.Module): 
    def __init__(self):
        self.iterCount = 0
        super(Discriminator, self).__init__() 
        self.main = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2,padding=1, bias = False),
            nn.LeakyReLU(0.2, inplace = True), 
            
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(128, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), 
            #nn.Sigmoid()
        )

    def forward(self, input): 
        output = self.main(input)
        return output.view(-1) #FLattening the result 

if reloadModels:
    gen = torch.load(modelFolder + 'generator_wg.pt')
    gen = gen.to(device)
    disc = torch.load(modelFolder + 'discriminator_wg.pt')
    disc = disc.to(device)
else:
    gen= Generator().to(device)
    gen.apply(weights_init) 
    print('Generator initialised')
    disc = Discriminator().to(device)
    disc.apply(weights_init)
    print('Discriminator initialised')
    
print(f' Models trained upto {gen.iterCount} epochs')

optimizerD = optim.RMSprop(disc.parameters(), lr = 0.00005)
optimizerG = optim.RMSprop(gen.parameters(), lr = 0.00005)

startFrom = gen.iterCount
for epoch in range(startFrom,nIter):
    disc.iterCount = disc.iterCount+1 
    gen.iterCount = gen.iterCount+1
    
    for i, data in enumerate(dataloader, 0):      
      for _ in range(3):
        for p in disc.parameters():
          p.requires_grad = True             
        for p in disc.parameters():
          p.data.clamp_(-0.01,0.01) 

        disc.zero_grad()
        
        # 1.a Training the discriminator with a real image of the dataset
        real, _ = data[0]
        input = Variable(real).to(device) 
        #target = Variable(torch.ones(input.size()[0])).to(device) 
        output = disc(input).to(device) 
        errD_real = -torch.mean(output)
        errD_real.backward()
        
        # 1.b Training the discriminator with a fake image generated by the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)).to(device)
        fake = gen(noise).to(device) 
        #target = Variable(torch.zeros(input.size()[0])).to(device)
        output = disc(fake.detach()).to(device) 
        errD_fake = torch.mean(output)
        errD_fake.backward()
        
        # Backpropagating the total error
        errD = errD_real - errD_fake
        #errD.backward()
        optimizerD.step()
           
        
      # 2 Training the generator
      for p in disc.parameters():
          p.requires_grad = False # to avoid computation
      gen.zero_grad()
      noise = Variable(torch.randn(input.size()[0], 100, 1, 1)).to(device)
      fake = gen(noise).to(device) 
      #target = Variable(torch.ones(input.size()[0])).to(device) 
      output = disc(fake).to(device) 
      errG = -torch.mean(output)
      errG.backward(retain_graph=True)
      optimizerG.step() 
        
      print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, nIter, i, len(dataloader), errD.item(), errG.item()))
        
            #print(errG.item())
    torch.save(gen,modelFolder+ 'generator_wg.pt')
    torch.save(disc,modelFolder+ 'discriminator_wg.pt')
        
    #3 Printing output and saving images and models
    
    if epoch % 1 == 0:
        Vutil.save_image(real, '%s/real_samples.png' % resultsFolder, normalize = True)
        fake = gen(noise)
        Vutil.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (resultsFolder, epoch), normalize = True)
