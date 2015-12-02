--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

paths.dofile('modelGenerator/modelGen.lua')
----------------------------------------------------------------------
local freeParams = function(m)
   local list = m:listModules()
   local p = 0
   for k,v in pairs(list) do
      p = p + (v.weight and v.weight:nElement() or 0)
      p = p + (v.bias and v.bias:nElement() or 0)
   end
   return p
end
----------------------------------------------------------------------
if opt.network ~= '' then
  print('<trainer> reloading previously trained network: ' .. opt.network)
  local tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
  print('Discriminator network:')
  print(model_D)
  print('Generator network:')
  print(model_G)
else
   local function weights_init(m)
      local name = torch.type(m)
      if name:find('Convolution') then
         m.weight:normal(0.02)
         m.bias:fill(0)
      elseif name:find('BatchNormalization') then
         if m.weight then m.weight:normal(1.0, 0.02) end
         if m.bias then m.bias:fill(0) end
      end
   end

   local nc = 3
   local nz = 100
   local ndf = 128
   local ngf = 128
   model_G = nn.Sequential()
   -- input is Z, going into a convolution
   model_G:add(nn.View(nz, 1, 1):setNumInputDims(3))
   model_G:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4)) -- 4x4 full-convolution initially
   model_G:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
   -- state size: (ngf*8) x 4 x 4
   model_G:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
   model_G:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
   -- state size: (ngf*4) x 8 x 8
   model_G:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
   model_G:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
   -- state size: (ngf*2) x 16 x 16
   model_G:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
   model_G:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
   -- state size: (ngf) x 32 x 32
   model_G:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
   model_G:add(nn.Tanh())
   -- state size: (nc) x 64 x 64

   model_G:apply(weights_init)
   ----------------------------------------------------------------------------
   model_D = nn.Sequential()

   -- input is (nc) x 64 x 64
   model_D:add(nn.SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
   model_D:add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf) x 32 x 32
   model_D:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
   model_D:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*2) x 16 x 16
   model_D:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
   model_D:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*4) x 8 x 8
   model_D:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
   model_D:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*8) x 4 x 4
   model_D:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4))
   model_D:add(nn.Sigmoid())
   -- state size: 1 x 1 x 1
   model_D:add(nn.View(1):setNumInputDims(3))
   -- state size: 1

   model_D:apply(weights_init)
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

model_D:cuda()
model_G:cuda()
criterion:cuda()


-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

print('\nNumber of free parameters in D: ' .. freeParams(model_D))
print('Number of free parameters in G: ' .. freeParams(model_G) .. '\n')
