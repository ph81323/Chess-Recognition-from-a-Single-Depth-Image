require 'nn'
require 'cunn'
require 'cudnn'

local net = nn.Sequential()

--conv1
net:add(nn.VolumetricConvolution(1,32,5,5,5,2,2,2,1,1,1))
--net:add(nn.LeakyReLU(0.1,true))
net:add(nn.ReLU(true))

net:add(nn.VolumetricDropout(0.2))

--conv2
net:add(nn.VolumetricConvolution(32,32,3,3,3,1,1,1))
--net:add(nn.LeakyReLU(0.1,true))
net:add(nn.ReLU(true))
--pool1
net:add(nn.VolumetricMaxPooling(2,2,2))
net:add(nn.VolumetricDropout(0.3))


--net:add(nn.View(-1, 11*11*23*32))
--net:add(nn.Linear(11*11*23*32,128))

--flat
net:add(nn.View(-1, 11*11*23*32))
--fc1
net:add(nn.Linear(11*11*23*32,128))
net:add(nn.ReLU(true))
net:add(nn.Dropout(0.4))

--net:add(nn.Linear(2048,128))
--fc2
net:add(nn.Linear(128,6))
--net:add(nn.Linear(128,2))
net:cuda()
return net
