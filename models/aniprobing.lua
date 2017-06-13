require 'nn'
require 'cunn'
require 'cudnn'

local net = nn.Sequential()
    
-- conv + batchnorm + relu
-- batchnorm is from cudnn R4, VERY useful for training deep net
local function Block3D(...)
    local arg = {...}
    net:add(cudnn.VolumetricConvolution(...))
    net:add(cudnn.VolumetricBatchNormalization(arg[2]))
    net:add(cudnn.ReLU(true))
    return net
end

local function Block2D(...)
    local arg = {...}
    net:add(cudnn.SpatialConvolution(...))
    net:add(cudnn.SpatialBatchNormalization(arg[2]))
    net:add(cudnn.ReLU(true))
    return net
end

-- Anisotropic probing layers
Block3D(1,5,100,1,1,100,1,1)
net:add(nn.View(1,5,50,50))
Block3D(1,5,5,1,1,5,1,1)
net:add(nn.View(1,5,50,50))
Block3D(1,1,5,1,1,5,1,1)
net:add(nn.View(1,50,50))

--net:add(nn.View(-1,50*50))
--net:add(nn.Linear(50*50,512))
--net:add(nn.Linear(512,128))
--net:add(nn.Linear(128,6))

-- 2D NIN
Block2D(1,128,5,5,1,1,3,3)
Block2D(128,96,1,1)
Block2D(96,96,1,1)
net:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
net:add(nn.Dropout(0.5))

Block2D(96,128,5,5,1,1,2,2)
Block2D(128,96,1,1)
Block2D(96,96,1,1)
net:add(cudnn.SpatialAveragePooling(3,3,2,2):ceil())
net:add(nn.Dropout(0.5))

Block2D(96,128,3,3,1,1,1,1)
Block2D(128,128,1,1)
Block2D(128,6,1,1)
net:add(cudnn.SpatialAveragePooling(13,13,1,1):ceil())
net:add(nn.View(-1,6))

local function MSRinit(net)
    local function init3d(name)
        for k,v in pairs(net:findModules(name)) do
            local n = v.kT*v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            v.bias:zero()
        end
    end
    local function init2d(name)
        for k,v in pairs(net:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            v.bias:zero()
        end
    end
    init3d'VolumetricConvolution'
    init2d'SpatialConvolution'
    return net
end

MSRinit(net)

net:cuda():forward(torch.randn(10,1,100,50,50):cuda())

return net
