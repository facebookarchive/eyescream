require 'torch'
require 'optim'
require 'pl'
require 'paths'
require 'image'

local adversarial = {}

-- training function
function adversarial.train(dataset, N)
  epoch = epoch or 1
  local N = N or dataset:size()
  local time = sys.clock()
  local dataBatchSize = opt.batchSize / 2

  local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local targets = torch.Tensor(opt.batchSize)
  local noise_inputs 
  if type(opt.noiseDim) == 'number' then
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)
  else
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  end
  local cond_inputs1
  local cond_inputs2
  if type(opt.condDim1) == 'number' then
    cond_inputs1 = torch.Tensor(opt.batchSize, opt.condDim1)
  else
    cond_inputs1 = torch.Tensor(opt.batchSize, opt.condDim1[1], opt.condDim1[2], opt.condDim1[3])
  end
  if type(opt.condDim2) == 'number' then
    cond_inputs2 = torch.Tensor(opt.batchSize, opt.condDim2)
  else
    cond_inputs2 = torch.Tensor(opt.batchSize, opt.condDim2[1], opt.condDim2[2], opt.condDim2[3])
  end

  -- do one epoch
  print('\n<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')
  for t = 1,N,dataBatchSize*opt.K do 

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients

      --  forward pass
      local outputs = model_D:forward({inputs, cond_inputs1, cond_inputs2})
      local f = criterion:forward(outputs, targets)

      -- backward pass 
      local df_do = criterion:backward(outputs, targets)
      model_D:backward({inputs, cond_inputs1, cond_inputs2}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
      end
      -- update confusion (add 1 since classes are binary)
      for i = 1,opt.batchSize do
        local c
        if outputs[i][1] > 0.5 then c = 2 else c = 1 end
        confusion:add(c, targets[i]+1)
      end

      return f,gradParameters_D
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = function(x)
      collectgarbage()
      if x ~= parameters_G then -- get new parameters
        parameters_G:copy(x)
      end
      
      gradParameters_G:zero() -- reset gradients

      -- forward pass
      local samples = model_G:forward({noise_inputs, cond_inputs1, cond_inputs2})
      local outputs = model_D:forward({samples, cond_inputs1, cond_inputs2})
      local f = criterion:forward(outputs, targets)

      --  backward pass
      local df_samples = criterion:backward(outputs, targets)
      model_D:backward({samples, cond_inputs1, cond_inputs2}, df_samples)
      local df_do = model_D.gradInput[1]
      model_G:backward({noise_inputs, cond_inputs1, cond_inputs2}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_G:add( sign(parameters_G):mul(opt.coefL1) + parameters_G:clone():mul(opt.coefL2) )
      end

      return f,gradParameters_G
    end

    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    -- Get half a minibatch of real, half fake
    for k=1,opt.K do
      -- (1.1) Real data 
      local k = 1
      for i = t,math.min(t+dataBatchSize-1,dataset:size()) do
        -- load new sample
        local idx = math.random(dataset:size())
        local sample = dataset[idx]
        inputs[k] = sample[1]:clone()
        cond_inputs1[k] = sample[2]:clone()
        cond_inputs2[k] = sample[3]:clone()
        k = k + 1
      end
      targets[{{1,dataBatchSize}}]:fill(1)
      -- (1.2) Sampled data
      noise_inputs:uniform(-1, 1)
      for i = dataBatchSize+1,opt.batchSize do
        local idx = math.random(dataset:size())
        local sample = dataset[idx]
        cond_inputs1[i] = sample[2]:clone()
        cond_inputs2[i] = sample[3]:clone()
      end
      local samples = model_G:forward({noise_inputs[{{dataBatchSize+1,opt.batchSize}}], cond_inputs1[{{dataBatchSize+1,opt.batchSize}}], cond_inputs2[{{dataBatchSize+1,opt.batchSize}}]})
      for i = 1, dataBatchSize do
        inputs[k] = samples[i]:clone()
        k = k + 1
      end
      targets[{{dataBatchSize+1,opt.batchSize}}]:fill(0)

      optim.sgd(fevalD, parameters_D, sgdState_D)
    end -- end for K

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    noise_inputs:uniform(-1, 1)
    for i = 1,opt.batchSize do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      cond_inputs1[i] = sample[2]:clone()
      cond_inputs2[i] = sample[3]:clone()
    end
    targets:fill(1)
    optim.sgd(fevalG, parameters_G, sgdState_G)

    -- disp progress
    xlua.progress(t, N)
  end -- end for loop over dataset

  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()

  -- save/log current net
  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, 'conditional_adversarial.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, {D = model_D, G = model_G, E = model_E, opt = opt})
  end

  -- next epoch
  epoch = epoch + 1
end

-- test function
function adversarial.test(dataset, N)
  local time = sys.clock()
  local N = N or dataset:size()

  local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local targets = torch.Tensor(opt.batchSize)
  local noise_inputs 
  if type(opt.noiseDim) == 'number' then
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)
  else
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  end
  local cond_inputs1
  local cond_inputs2
  if type(opt.condDim1) == 'number' then
    cond_inputs1 = torch.Tensor(opt.batchSize, opt.condDim1)
  else
    cond_inputs1 = torch.Tensor(opt.batchSize, opt.condDim1[1], opt.condDim1[2], opt.condDim1[3])
  end
  if type(opt.condDim2) == 'number' then
    cond_inputs2 = torch.Tensor(opt.batchSize, opt.condDim2)
  else
    cond_inputs2 = torch.Tensor(opt.batchSize, opt.condDim2[1], opt.condDim2[2], opt.condDim2[3])
  end


  print('\n<trainer> on testing set:')
  for t = 1,N,opt.batchSize do
    -- display progress
    xlua.progress(t, N)

    ----------------------------------------------------------------------
    -- (1) Real data
    local targets = torch.ones(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      inputs[k] = sample[1]:clone()
      cond_inputs1[k] = sample[2]:clone()
      cond_inputs2[k] = sample[3]:clone()
      k = k + 1
    end
    local preds = model_D:forward({inputs, cond_inputs1, cond_inputs2}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end

    ----------------------------------------------------------------------
    -- (2) Generated data (don't need this really, since no 'validation' generations)
    noise_inputs:uniform(-1, 1)
    local c = 1
    for i = 1,opt.batchSize do
      sample = dataset[math.random(dataset:size())]
      cond_inputs1[i] = sample[2]:clone()
      cond_inputs2[i] = sample[3]:clone()
    end
    local samples = model_G:forward({noise_inputs, cond_inputs1, cond_inputs2})
    local targets = torch.zeros(opt.batchSize)
    local preds = model_D:forward({samples, cond_inputs1, cond_inputs2}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()

  local samples = model_G.output
  local to_plot = {}
  for i=1,opt.batchSize do
    to_plot[i] = samples[i]:float()
  end
  local fname = paths.concat(opt.save, 'epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})
  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end
  return cond_inputs
end

return adversarial
