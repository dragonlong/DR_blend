import torch
import torch.nn as nn
import torch.nn.functional as F
"""
For nonlinearity, we use leaky (0.01) rectifier units following each convolutional layer. 
The networks are trained with Nesterov momentum with fixed schedule over 250 epochs. For the nets 
on 256 and 128 pixel images, we stop training after 200 epochs. L2 weight decay with factor 0.0005
are applied to all layers. 
The problem is treated as a regression problem, the loss function is mean squared error.
"""
class c_512_5_3_32(nn.Module):
    def __init__(self):
        super(c_512_5_3_32, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=(3,3),stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=(3,3), stride=2)
        self.dropout = nn.Dropout(p= 0.5)
        self.leaky_relu = nn.LeakyReLU(0.01)
#         # net 4
#         self.conv1_1 = nn.Conv2d(3, 32, kernel_size=(4,4), stride=2)
#         self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(4,4), padding=2) #border_mode=None

#         self.conv2_1 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=2)
#         self.conv2_2 = nn.Conv2d(64, 64, kernel_size=(4, 4), padding=2) #border_mode=None
#         self.conv2_3 = nn.Conv2d(64, 64, kernel_size=(4, 4))

#         self.conv3_1 = nn.Conv2d(64, 128, kernel_size=(4, 4), padding=2) #border_mode=None
#         self.conv3_2 = nn.Conv2d(128, 128, kernel_size=(4, 4))
#         self.conv3_3 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=2) #border_mode=None

#         self.conv4_1 = nn.Conv2d(128, 256, kernel_size=(4, 4), padding=2) #border_mode=None
#         self.conv4_2 = nn.Conv2d(256, 256, kernel_size=(4, 4))
#         self.conv4_3 = nn.Conv2d(256, 256, kernel_size=(4, 4), padding=2) #border_mode=None

#         self.conv5_1 = nn.Conv2d(256, 512, kernel_size=(4, 4), padding=3)

        # net 5
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=(5,5), stride=2)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=2) #border_mode=None

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=(5,5), stride=2)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=2) #border_mode=None
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=(3, 3))

        self.conv3_1 = nn.Conv2d(64, 128,  kernel_size=(3, 3), padding=2) #border_mode=None
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=2) #border_mode=None

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=2) #border_mode=None
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=(3, 3))
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=2) #border_mode=None

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=(3, 3))
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3))
        
        self.fc_1 = nn.Linear(256, 1)
        #x = F.adaptive_avg_pool2d(x, (1, 1))

        #self.ap = nn.AvgPool2d(3, stride=(2, 2))
        '''
        self.fc_1 = nn.Linear(1024)

        self.fc_2 = nn.Linear(1024)

        self.mo = nn.MaxOut(1024, 1024, 2)

        self.RMSPool = torch.sqrt(nn.AvgPool2d(torch.sqrt(x), kernel_size=3, stride=(2,2)) + 1e-12)
        '''
    def forward(self, x):

        x = self.max_pool(self.leaky_relu(self.conv1_2(self.leaky_relu(self.conv1_1(x)))))
        #print(x.size())
        x = self.leaky_relu(self.conv2_3(self.leaky_relu(self.conv2_2(self.leaky_relu(self.conv2_1(x))))))
        #print(x.size())
        #x = self.max_pool(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        x = self.leaky_relu(self.conv3_3(self.leaky_relu(self.conv3_2(self.leaky_relu(self.conv3_1(x))))))
        #print(x.size())
        x = self.leaky_relu(self.conv4_3(self.leaky_relu(self.conv4_2(self.leaky_relu(self.conv4_1(x))))))
        #print(x.size())
        last_conv = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        ram_x = x
        #x = (self.conv5_1(x))
        #print(x.size())
        #x = torch.sqrt(self.ap(torch.sqrt(x)) + 1e-12)  #RMSPool
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        #print(x.size())
        '''
        x = self.mo(self.fc_1(self.dropout(x)))
        print(x.size())
        x = self.mo(self.fc_2(self.dropout(x)))
        print(x.size())
        '''
        return x, ram_x, last_conv


# class Maxout(nn.Module):

#     def __init__(self, d_in, d_out, pool_size):
#         super().__init__()
#         self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
#         self.lin = nn.Linear(d_in, d_out * pool_size)
#     def forward(self, inputs):
#         shape = list(inputs.size())
#         shape[-1] = self.d_out
#         shape.append(self.pool_size)
#         max_dim = len(shape) - 1
#         out = self.lin(inputs)
#         m, i = out.view(*shape).max(max_dim)
#         return m
if __name__ == '__main__':
    dd = torch.randn(1, 3, 512, 512)
    dd = torch.autograd.Variable(dd)
    model = c_512_5_3_32()
    y = model(dd)
    print (y.size())
