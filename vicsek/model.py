import torch
import torch.nn as nn

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
        