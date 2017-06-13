require 'nn'
require 'cunn'
require 'cudnn'

local net = nn.Sequential()

--Block(1,48,6,6,6,2,2,2)
net:add(nn.VolumetricConvolution(1,48,6,6,6,2,2,2))
net:add(cudnn.VolumetricBatchNormalization(48))
net:add(cudnn.ReLU(true))
--Block(48,48,1,1,1)
net:add(nn.VolumetricConvolution(48,48,1,1,1))
net:add(cudnn.VolumetricBatchNormalization(48))
net:add(cudnn.ReLU(true))
--Block(48,48,1,1,1)
net:add(nn.VolumetricConvolution(48,48,1,1,1))
net:add(cudnn.VolumetricBatchNormalization(48))
net:add(cudnn.ReLU(true))
net:add(nn.Dropout(0.2))

--Block(48,96,5,5,5,2,2,2)
net:add(nn.VolumetricConvolution(48,96,5,5,5,2,2,2))
net:add(nn.LeakyReLU(0.1,true))
net:add(nn.VolumetricMaxPooling(2,2,2))
net:add(nn.VolumetricDropout(0.3))

net:add(nn.View(-1, 5*5*11*96))
net:add(nn.Linear(5*5*11*96,512))
net:add(cudnn.ReLU(true))
net:add(nn.Dropout(0.5))

net:add(nn.Linear(512,40))
net:add(nn.Linear(128,6))
return net
