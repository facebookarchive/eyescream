--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'torch'
require 'optim'
require 'paths'

local adversarial = {}

local unpack = unpack and unpack or table.unpack

-- reusable buffers
local targets      = torch.CudaTensor(opt.batchSize)
local inputs       = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))               -- original full-res image - low res image
local cond_inputs  = torch.CudaTensor(opt.batchSize, unpack(opt.condDim))  -- low res image blown up and differenced from original
local noise_inputs = torch.CudaTensor(opt.batchSize, unpack(opt.noiseDim)) -- pure noise
local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()

-- training function
function adversarial.train(inputs_all, inputs_all2)
   local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
   local err_G, err_D = -1, -1

   -- inputs_all = {diff, label, coarse, fine}
   inputs:copy(inputs_all[1])
   cond_inputs:copy(inputs_all[3])

   -- create closure to evaluate f(X) and df/dX of discriminator
   local fevalD = function(x)
      collectgarbage()
      gradParameters_D:zero() -- reset gradients

      local finputs = {inputs, cond_inputs}
      if opt.scratch == 1 then
         finputs = inputs
      end

      --  forward pass
      local outputs = model_D:forward(finputs)
      err_D = criterion:forward(outputs, targets)

      -- backward pass
      local df_do = criterion:backward(outputs, targets)
      model_D:backward(finputs, df_do)

      -- update confusion (add 1 since classes are binary)
      outputs[outputs:gt(0.5)] = 2
      outputs[outputs:le(0.5)] = 1
      confusion:batchAdd(outputs, targets:clone():add(1))

      return err_D,gradParameters_D
   end
   ----------------------------------------------------------------------
   -- create closure to evaluate f(X) and df/dX of generator
   local fevalG = function(x)
      collectgarbage()
      gradParameters_G:zero() -- reset gradients
      local finputsG = {noise_inputs, cond_inputs}
      if opt.scratch == 1 then
         finputsG = noise_inputs
      end
      -- forward pass
      local hallucinations = model_G:forward(finputsG)
      local finputsD = {hallucinations, cond_inputs}
      if opt.scratch == 1 then
         finputsD = hallucinations
      end
      local outputs = model_D:forward(finputsD)
      err_G = criterion:forward(outputs, targets)

      --  backward pass
      local df_hallucinations = criterion:backward(outputs, targets)
      model_D:backward(finputsD, df_hallucinations)
      local df_do = model_D.modules[1].gradInput[1]
      if opt.scratch == 1 then
         df_do = model_D.gradInput
      end
      model_G:backward(finputsG, df_do)

      return err_G,gradParameters_G
   end
   ----------------------------------------------------------------------
   -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
   assert (opt.batchSize % 2 == 0)
   -- (1.1) Real data is in {inputs, cond_inputs}
   targets:fill(1)
   -- (1.2) Sampled data
   noise_inputs:uniform(-1, 1)
   local inps = {noise_inputs, cond_inputs}
   if opt.scratch == 1 then -- no scale conditioning if training from scratch
      inps = noise_inputs
   end

   local hallucinations = model_G:forward(inps)
   assert(hallucinations:size(1) == opt.batchSize)
   assert(hallucinations:size(2) == 3)
   assert(hallucinations:nElement() == inputs:nElement())
   -- print(#hallucinations)
   -- print(#inputs)
   inputs:narrow(1, 1, opt.batchSize / 2):copy(hallucinations:narrow(1, 1, opt.batchSize / 2))
   targets:narrow(1, 1, opt.batchSize / 2):fill(0)
   -- evaluate inputs and get the err for G and D separately
   local optimizeG = false
   local optimizeD = false
   local err_R, err_F
   do
      local margin = 0.3

      local finputs = {inputs, cond_inputs}
      if opt.scratch == 1 then
         finputs = inputs
      end

      local outputs = model_D:forward(finputs)
      assert(targets:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2):min() == 1)
      err_F = criterion:forward(outputs:narrow(1, 1, opt.batchSize / 2), targets:narrow(1, 1, opt.batchSize / 2))
      err_R = criterion:forward(outputs:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2), targets:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2))
      if err_F > err_R + margin then
         optimizeG = false; optimizeD = true;
      elseif err_F > err_R  and err_F <= err_R + margin then optimizeG = true; optimizeD = false;
      elseif err_F <= err_R then
         optimizeG = true; optimizeD = false;
      end
      if err_R > 0.7 then optimizeD = true; end
   end
   if optimizeD then
      optim.sgd(fevalD, parameters_D, sgdState_D)
   end
   ----------------------------------------------------------------------
   -- (2) Update G network: maximize log(D(G(z)))
   noise_inputs:uniform(-1, 1)
   targets:fill(1)
   cond_inputs:copy(inputs_all2[3])
   if optimizeG then
      optim.sgd(fevalG, parameters_G, sgdState_G)
   end
   batchNumber = batchNumber + 1
   cutorch.synchronize(); collectgarbage();
   -- xlua.progress(batchNumber, opt.epochSize)
   print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f Err_G %.4f Err_D %.4f Err_R %.4f Err_F %.4f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime, err_G, err_D, err_R, err_F))
   dataTimer:reset()
end

-- test function
function adversarial.test(inputs_all)
   -- (1) Real data
   targets:fill(1)
   inputs:copy(inputs_all[1])
   cond_inputs:copy(inputs_all[3])
   local finputs = {inputs, cond_inputs}
   if opt.scratch == 1 then
      finputs = inputs
   end

   local outputs = model_D:forward(finputs) -- get predictions from D

   -- add to confusion matrix
   outputs[outputs:gt(0.5)] = 2
   outputs[outputs:le(0.5)] = 1
   confusion:batchAdd(outputs, targets:clone():add(1))
   ----------------------------------------------------------------------
   -- (2) Generated data
   noise_inputs:uniform(-1, 1)
   local finputsG = {noise_inputs, cond_inputs}
   if opt.scratch == 1 then
      finputsG = noise_inputs
   end
   local samples = model_G:forward(finputsG)
   targets:fill(0)
   local finputsD = {samples, cond_inputs}
   if opt.scratch == 1 then
      finputsD = samples
   end
   local outputs = model_D:forward(finputsD)
end

return adversarial
