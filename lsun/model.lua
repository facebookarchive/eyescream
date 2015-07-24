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
elseif opt.model == 'small' then
   local nplanes = 64
   model_D = nn.Sequential()
   model_D:add(nn.CAddTable())
   model_D:add(nn.SpatialConvolution(3, nplanes, 5, 5)) --28 x 28
   model_D:add(nn.ReLU())
   model_D:add(nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2))
   local sz = math.floor( ( (opt.fineSize - 5 + 1) - 5) / 2 + 1)
   model_D:add(nn.View(nplanes*sz*sz))
   model_D:add(nn.ReLU())
   model_D:add(nn.Linear(nplanes*sz*sz, 1))
   model_D:add(nn.Sigmoid())
   local nplanes = 128
   model_G = nn.Sequential()
   model_G:add(nn.JoinTable(2, 2))
   model_G:add(cudnn.SpatialConvolutionUpsample(3+1, nplanes, 7, 7, 1))
   model_G:add(nn.ReLU())
   model_G:add(cudnn.SpatialConvolutionUpsample(nplanes, nplanes, 7, 7, 1))
   model_G:add(nn.ReLU())
   model_G:add(cudnn.SpatialConvolutionUpsample(nplanes, 3, 5, 5, 1))
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
elseif opt.model == 'large' then
   require 'fbcunn'
   print('Generator network (good):')
   desc_G = '___JT22___C_4_64_g1_7x7___R__BN___C_64_368_g4_7x7___R__BN___SDrop 0.5___C_368_128_g4_7x7___R__BN___P_LPOut_2___C_64_224_g2_5x5___R__BN___SDrop 0.5___C_224_3_g1_7x7__BNA'
   model_G = nn.Sequential()
   model_G:add(nn.JoinTable(2, 2))
   model_G:add(cudnn.SpatialConvolutionUpsample(3+1, 64, 7, 7, 1, 1)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(64, nil, nil, false))
   model_G:add(cudnn.SpatialConvolutionUpsample(64, 368, 7, 7, 1, 4)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(368, nil, nil, false))
   model_G:add(nn.SpatialDropout(0.5))
   model_G:add(cudnn.SpatialConvolutionUpsample(368, 128, 7, 7, 1, 4)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(128, nil, nil, false))
   model_G:add(nn.FeatureLPPooling(2,2,2,true))
   model_G:add(cudnn.SpatialConvolutionUpsample(64, 224, 5, 5, 1, 2)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(224, nil, nil, false))
   model_G:add(nn.SpatialDropout(0.5))
   model_G:add(cudnn.SpatialConvolutionUpsample(224, 3, 7, 7, 1, 1))
   model_G:add(nn.SpatialBatchNormalization(3, nil, nil, false))
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
   print(desc_G)

   desc_D = '___CAdd___C_3_48_g1_3x3___R___C_48_448_g4_5x5___R___C_448_416_g16_7x7___R___V_166400___L 166400_1___Sig'
   model_D = nn.Sequential()
   model_D:add(nn.CAddTable())
   model_D:add(cudnn.SpatialConvolution(3, 48, 3, 3))
   model_D:add(cudnn.ReLU(true))
   model_D:add(cudnn.SpatialConvolution(48, 448, 5, 5, 1, 1, 0, 0, 4))
   model_D:add(cudnn.ReLU(true))
   model_D:add(cudnn.SpatialConvolution(448, 416, 7, 7, 1, 1, 0, 0, 16))
   model_D:add(cudnn.ReLU())
   model_D:cuda()
   local dummy_input = torch.zeros(opt.batchSize, 3, opt.fineSize, opt.fineSize):cuda()
   local out = model_D:forward({dummy_input, dummy_input})
   local nElem = out:nElement() / opt.batchSize
   model_D:add(nn.View(nElem):setNumInputDims(3))
   model_D:add(nn.Linear(nElem, 1))
   model_D:add(nn.Sigmoid())
   model_D:cuda()
   print(desc_D)
elseif opt.model == 'autogen' then
   -- define G network to train
   print('Generator network:')
   model_G,desc_G = generateModelG(3,5,128,512,3,7, 'mixed', 0, 4, 2, true)
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
   print(desc_G)
   print(model_G)
   local trygen = 1
   local poolType = 'none'
   -- if torch.random(1,2) == 1 then poolType = 'none' end
   repeat
      trygen = trygen + 1
      if trygen == 500 then error('Could not find a good D model') end
      -- define D network to train
      print('Discriminator network:')
      model_D,desc_D = generateModelD(2,6,64,512,3,7, poolType, 0, 4, 2)
      print(desc_D)
      print(model_D)
   until (freeParams(model_D) < freeParams(model_G))
      and (freeParams(model_D) > freeParams(model_G) / 10)
elseif opt.model == 'full' or opt.model == 'fullgen' then
   local nhid = 1024
   local nhidlayers = 2
   local batchnorm = 1 -- disabled
   if opt.model == 'fullgen' then
      nhidlayers = torch.random(1,5)
      nhid = torch.random(8, 128) * 16
      batchnorm = torch.random(1,2)
   end
   desc_G = ''
   model_G = nn.Sequential()
   model_G:add(nn.JoinTable(2, 2))
   desc_G = desc_G .. '___JT22'
   model_G:add(nn.View(4 * opt.fineSize * opt.fineSize):setNumInputDims(3))
   desc_G = desc_G .. '___V_' .. 4 * opt.fineSize * opt.fineSize
   model_G:add(nn.Linear(4 * opt.fineSize * opt.fineSize, nhid)):add(nn.ReLU())
   desc_G = desc_G .. '___L ' .. 4 * opt.fineSize * opt.fineSize .. '_' .. nhid
   desc_G = desc_G .. '__R'
   if batchnorm == 2 then
      model_G:add(nn.BatchNormalization(nhid), nil, nil, true)
      desc_G = desc_G .. '__BNA'
   end
   model_G:add(nn.Dropout(0.5))
   desc_G = desc_G .. '__Drop' .. 0.5
   for i=1,nhidlayers do
      model_G:add(nn.Linear(nhid, nhid)):add(nn.ReLU())
      desc_G = desc_G .. '___L ' .. nhid .. '_' .. nhid
      desc_G = desc_G .. '__R'
      if batchnorm == 2 then
         model_G:add(nn.BatchNormalization(nhid), nil, nil, true)
         desc_G = desc_G .. '__BNA'
      end
      model_G:add(nn.Dropout(0.5))
      desc_G = desc_G .. '__Drop' .. 0.5
   end
   model_G:add(nn.Linear(nhid, opt.geometry[1]*opt.geometry[2]*opt.geometry[3]))
   desc_G = desc_G .. '___L ' .. nhid .. '_' .. opt.geometry[1]*opt.geometry[2]*opt.geometry[3]
   if batchnorm == 2 then
      model_G:add(nn.BatchNormalization(opt.geometry[1]*opt.geometry[2]*opt.geometry[3]))
      desc_G = desc_G .. '__BNA'
   end
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
   desc_G = desc_G .. '___V_' .. opt.geometry[1] .. '_' ..  opt.geometry[2] .. '_' ..  opt.geometry[3]
   print(desc_G)
   print(model_G)

   nhid = nhid / 2
   desc_D = ''
   model_D = nn.Sequential()
   model_D:add(nn.CAddTable())
   desc_D = desc_D .. '___CAdd'
   model_D:add(nn.View(opt.geometry[1]* opt.geometry[2]* opt.geometry[3]))
   desc_D = desc_D .. '___V_' .. opt.geometry[1]* opt.geometry[2]* opt.geometry[3]
   model_D:add(nn.Linear(opt.geometry[1]* opt.geometry[2]* opt.geometry[3], nhid)):add(nn.ReLU())
   desc_D = desc_D .. '___L ' .. opt.geometry[1]* opt.geometry[2]* opt.geometry[3] .. '_' .. nhid
   desc_D = desc_D .. '__R'
   for i=1,nhidlayers do
      model_D:add(nn.Linear(nhid, nhid)):add(nn.ReLU())
      desc_D = desc_D .. '___L ' .. nhid .. '_' .. nhid
      desc_D = desc_D .. '__R'
      model_D:add(nn.Dropout(0.5))
      desc_D = desc_D .. '__Drop' .. 0.5
   end
   model_D:add(nn.Linear(nhid, 1))
   desc_D = desc_D .. '___L ' .. nhid .. '_' .. 1
   model_D:add(nn.Sigmoid())
   desc_D = desc_D .. '__Sig'
   model_D:cuda()
   print(desc_D)
   print(model_D)
elseif opt.model == 'small_18' then
   assert(opt.scratch == 1) -- check that this is not conditional on a previous scale
   ----------------------------------------------------------------------
   local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]
   -- define D network to train
   local numhid = 600
   model_D = nn.Sequential()
   model_D:add(nn.View(input_sz):setNumInputDims(3))
   model_D:add(nn.Linear(input_sz, numhid))
   model_D:add(nn.ReLU())
   model_D:add(nn.Dropout())
   model_D:add(nn.Linear(numhid, numhid))
   model_D:add(nn.ReLU())
   model_D:add(nn.Dropout())
   model_D:add(nn.Linear(numhid, 1))
   model_D:add(nn.Sigmoid())
   ----------------------------------------------------------------------
   local noiseDim = opt.noiseDim[1] * opt.noiseDim[2] * opt.noiseDim[3]
   -- define G network to train
   local numhid = 600
   model_G = nn.Sequential()
   model_G:add(nn.View(noiseDim):setNumInputDims(3))
   model_G:add(nn.Linear(noiseDim, numhid))
   model_G:add(nn.ReLU())
   model_G:add(nn.Linear(numhid, numhid))
   model_G:add(nn.ReLU())
   model_G:add(nn.Linear(numhid, input_sz))
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
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
