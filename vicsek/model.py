import torch
# import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn
# import torch.nn.functional as F
# import pickle as pkl
# from torch_geometric.utils import dense_to_sparse
# import torch_geometric
# from torch_geometric.nn import GCNConv,GATConv
# from tqdm import tqdm
# import wandb
# import random

'0505动力学肯定还有优化空间，现在训练非常慢，而且直观上有很多不必要的设计'
def deconv(input_channels, output_channels):
    layer = nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )
    return layer

def conv(input_channels, output_channels, kernel_size, stride, dropout_rate=0):
    layer = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, 
                  stride = stride, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
#         nn.Dropout(dropout_rate)
    )
    return layer

class Encoder(nn.Module):
    def __init__(self, input_channels, kernel_size):
        super(Encoder, self).__init__()
        self.conv1 = conv(input_channels, 32, kernel_size=kernel_size, stride = 2)
        self.conv2 = conv(32, 64, kernel_size=kernel_size, stride = 2)
#         self.conv3 = conv(128, 256, kernel_size=kernel_size, stride = 2, dropout_rate = dropout_rate)
#         self.conv4 = conv(256, 512, kernel_size=kernel_size, stride = 2, dropout_rate = dropout_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.002/n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
#         out_conv3 = self.conv3(out_conv2)
#         out_conv4 = self.conv4(out_conv3)
        return out_conv1, out_conv2#, out_conv3, out_conv4 

class Tempro_Spacial(nn.Module):
    def __init__(self, time_range, kernel_size):
        super(Tempro_Spacial, self).__init__()
        self.spatial_filter = nn.Conv2d(1, 1, kernel_size = kernel_size, padding = 1, bias = False)   
        self.temporal_filter = nn.Conv2d(time_range, time_range, kernel_size = 1, padding = 0, bias = False)
        self.encoder = Encoder(time_range*2, kernel_size)
    def forward(self, x):
#         x.shape = b,t,l,l
        b,t,l1,l2 = x.shape
        x = x.reshape(b*t,1,l1,l2)
#         x = torch.transpose(torch.transpose(x,2,3),1,2) #b*t,1,l,l
        space_conv = self.spatial_filter(x)#b*t,1,l,l
        space_conv = space_conv.reshape(b,t,l1,l2)#b,t,l,l
#         print(space_conv.shape)
        temp_conv = self.temporal_filter(space_conv)#b,t,l,l
#         print(space_conv.shape, temp_conv.shape)
        out_conv1, out_conv2 = self.encoder(torch.cat([space_conv,temp_conv],dim=1))
        
        return out_conv1, out_conv2#, out_conv3, out_conv4 


class DynamicLearner(nn.Module):
    def __init__(self,time_range=5, time_range_out=10,L=64,input_dim=3,output_dim=2):
        super(DynamicLearner, self).__init__()
#         self.hidden_dim = hidden_dim
        
#         channel = 1
        self.time_range_out = time_range_out
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.time_range = time_range
        self.kernel_size = 3
        
        self.tempro_spacial1 = Tempro_Spacial(time_range,self.kernel_size)
        self.tempro_spacial2 = Tempro_Spacial(time_range,self.kernel_size)
        self.tempro_spacial3 = Tempro_Spacial(time_range,self.kernel_size)
        
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)
        
#         self.output_layer = nn.Sequential(
#             nn.Linear(hidden_dim,output_dim),
#         )
        self.output_layer = nn.Conv2d(16+input_dim, output_dim*time_range_out, kernel_size = self.kernel_size, padding = 1)  
    
        
    def forward(self, xx):
        'x.shape: b,t,l,l,f'
        b,t,l1,l2,f = xx.shape
        out_conv1_rho, out_conv2_rho = self.tempro_spacial1(xx[:,:,:,:,0])
        out_conv1_vx, out_conv2_vx = self.tempro_spacial2(xx[:,:,:,:,1])
        out_conv1_vy, out_conv2_vy = self.tempro_spacial3(xx[:,:,:,:,2]) #output = 64,64/128,64,64
        
        out_deconv1 = self.deconv2(out_conv2_rho + out_conv2_vx + out_conv2_vy)
        out_deconv0 = self.deconv1(out_conv1_rho + out_conv1_vx + out_conv1_vy + out_deconv1)#output=64,32,64,64
#         out_deconv3 = self.deconv3(out_conv1 + out_conv2)
#         print(out_deconv0.shape)
        x = torch.transpose(torch.transpose(xx[:,-1,:,:,:],3,2),2,1) #b,time_range_out,l,l,hidden_dim
        x = torch.cat([x, out_deconv0],dim=1)
        x = self.output_layer(x).reshape(b,self.time_range_out,self.output_dim,l1,l2)##b,time_range_out*output_dim,l,l
        x = torch.transpose(torch.transpose(x,2,3),3,4)
#         print(x.shape)
#         de
        return x



class Normalization(nn.Module):
    def __init__(self, group_size=2,input_dim=3, output_dim=3):
        super(Normalization, self).__init__()
        self.group_size=group_size
        kernel_size = group_size
        stride = group_size
        input_channels = input_dim
        output_channels = output_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, group_size, padding=1,padding_mode='circular'),
            nn.MaxPool2d(group_size),
#             nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, group_size, stride=(group_size,group_size)),
#             nn.ReLU()
#             nn.ConvTranspose2d(channel, 1, group_size, padding=1),
#             nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv(x)
#         x = relu(x)
        return x
    def Encoder(self,x):
        b,t,l,_,f = x.shape
        x = x.reshape(b*t,l,l,f)
        x = torch.transpose(torch.transpose(x,3,2),2,1)
#         print(x.shape)
#         de
        x = self.encoder(x)
        x = torch.transpose(torch.transpose(x,1,2),2,3)
        x = x.reshape(b,t,l//self.group_size,l//self.group_size,f)
        return x
    def Decoder(self,x):
        b,t,l,_,f = x.shape
        x = x.reshape(b*t,l,l,f)
        x = torch.transpose(torch.transpose(x,3,2),2,1)
#         print(x.shape)
        x = self.decoder(x)
#         print(x.shape)
        x = torch.transpose(torch.transpose(x,1,2),2,3)
#         print(x.shape)
        x = x.unsqueeze(1)
#         print(x.shape)
#         de
#         x = relu(x)
        return x


def deconv(input_channels, output_channels):
    layer = nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )
    return layer

def conv(input_channels, output_channels, kernel_size, stride, dropout_rate=0):
    layer = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, 
                  stride = stride, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )
    return layer

class Encoder(nn.Module):
    def __init__(self, input_channels, kernel_size):
        super(Encoder, self).__init__()
        self.conv1 = conv(input_channels, 8, kernel_size=kernel_size, stride = 2)
                
    def forward(self, x):
        out_conv1 = self.conv1(x)
        return out_conv1#, out_conv2#, out_conv3, out_conv4 


'0513'
class Tempro_Spacial(nn.Module):
    def __init__(self, time_range, kernel_size):
        super(Tempro_Spacial, self).__init__()
        self.time_range = time_range
        self.filter = nn.Conv2d(time_range, 1, kernel_size = kernel_size, padding = 1, bias = False)   
        self.temporal_filter = nn.Conv1d(1, 1, kernel_size = time_range, padding = 0, bias = False)
        self.encoder = Encoder(1, kernel_size)
    def forward(self, x):
#         x.shape = b,t,l,l
        b,t,l1,l2 = x.shape
        space_conv = self.filter(x)#b,1,l,l
#         print(space_conv.shape)
        out_conv1 = self.encoder(space_conv)#b,8,l,l
        return out_conv1
    
class DynamicLearner(nn.Module):
    def __init__(self,time_range=5, time_range_out=1,L=64,input_dim=3,output_dim=3):
        super(DynamicLearner, self).__init__()
#         self.hidden_dim = hidden_dim
        
#         channel = 1
        self.time_range_out = time_range_out
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.time_range = time_range
        self.kernel_size = 3
        
        self.tempro_spacial1 = Tempro_Spacial(time_range,self.kernel_size)
        self.tempro_spacial2 = Tempro_Spacial(time_range,self.kernel_size)
        self.tempro_spacial3 = Tempro_Spacial(time_range,self.kernel_size)
        
#         self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(8,4)
        self.output_layer = nn.Conv2d(4+input_dim, output_dim*time_range_out, kernel_size = self.kernel_size, padding = 1)  
    
        
    def forward(self, xx):
        'x.shape: b,t,l,l,f'
        b,t,l1,l2,f = xx.shape
        out_conv1_rho = self.tempro_spacial1(xx[:,:,:,:,0])
        out_conv1_vx = self.tempro_spacial2(xx[:,:,:,:,1])
        out_conv1_vy = self.tempro_spacial3(xx[:,:,:,:,2]) #output = 64,64/128,64,64
        
        out_deconv0 = self.deconv1(out_conv1_rho + out_conv1_vx + out_conv1_vy)#output=b,4,l,l
        
        x = torch.transpose(torch.transpose(xx[:,-1,:,:,:],3,2),2,1) #b,l,l,f --> b,l,f,l --> b,f,l,l
#         print(x[:5,:5,:5,:5])
        x = xx[:,-1,:,:,:].permute(0,3,1,2)
#         print(x[:5,:5,:5,:5])
#         de
        x = torch.cat([x, out_deconv0],dim=1)#b,f+4,l,l
        x = self.output_layer(x).reshape(b,self.time_range_out,self.output_dim,l1,l2)##b,1,3,l,l
        x = torch.transpose(torch.transpose(x,2,3),3,4)#b,1,l,l,3
#         print(x.shape)
#         de
        return x
        