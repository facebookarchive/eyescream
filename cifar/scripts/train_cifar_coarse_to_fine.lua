require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'datasets.coarse_to_fine_cifar10'
require 'pl'
require 'paths'
image_utils = require 'utils.image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'train.conditional_adversarial'
require 'layers.SpatialConvolutionUpsample'


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 5)           save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.02)        learning rate
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0.5)           momentum
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default -1)          gpu to run on (default cpu)
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --hidden_G         (default 64)         number of channels in hidden layers of G
  --hidden_D         (default 64)         number of channels in hidden layers of D
  --coarseSize       (default 16)          coarse scale
  --fineSize         (default 32)          fine scale
]]

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end
print(opt)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.noiseDim = {1, opt.fineSize, opt.fineSize}
classes = {'0','1'}
opt.geometry = {3, opt.fineSize, opt.fineSize}
opt.condDim = {3, opt.fineSize, opt.fineSize}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ----------------------------------------------------------------------
  -- define D network to train
  local nplanes = opt.hidden_D
  model_D = nn.Sequential()
  model_D:add(nn.CAddTable())
  model_D:add(nn.SpatialConvolution(3, nplanes, 5, 5)) --28 x 28
  model_D:add(nn.ReLU())
  model_D:add(nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2))
  local sz =math.floor( ( (opt.fineSize - 5 + 1) - 5) / 2 + 1)
  model_D:add(nn.Reshape(nplanes*sz*sz))
  model_D:add(nn.ReLU())
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(nplanes*sz*sz, 1))
  model_D:add(nn.Sigmoid())

  ----------------------------------------------------------------------
  -- define G network to train
  local nplanes = opt.hidden_G
  model_G = nn.Sequential()
  model_G:add(nn.JoinTable(2, 2))
  model_G:add(nn.SpatialConvolutionUpsample(3+1, nplanes, 7, 7, 1)) -- 3 color channels + conditional
  model_G:add(nn.ReLU())
  model_G:add(nn.SpatialConvolutionUpsample(nplanes, nplanes, 7, 7, 1)) -- 3 color channels + conditional
  model_G:add(nn.ReLU())
  model_G:add(nn.SpatialConvolutionUpsample(nplanes, 3, 5, 5, 1)) -- 3 color channels + conditional
  model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Evaluator network:')
print(model_G)

local nparams = 0
for i=1,#model_D.modules do
  if model_D.modules[i].weight ~= nil then
    nparams = nparams + model_D.modules[i].weight:nElement()
  end
end
print('\nNumber of free parameters in D: ' .. nparams)


local nparams = 0
for i=1,#model_G.modules do
  if model_G.modules[i].weight ~= nil then
    nparams = nparams + model_G.modules[i].weight:nElement()
  end
end
print('Number of free parameters in G: ' .. nparams .. '\n')

----------------------------------------------------------------------
-- get/create dataset
--
ntrain = 45000
nval = 5000

cifar.init(opt.fineSize, opt.coarseSize)

-- create training set and normalize
trainData = cifar.loadTrainSet(1, ntrain)
mean, std = image_utils.normalize(trainData.data)
trainData:makeFine()
trainData:makeCoarse()
trainData:makeDiff()

-- create validation set and normalize
valData = cifar.loadTrainSet(ntrain+1, ntrain+nval)
image_utils.normalize(valData.data, mean, std)
valData:makeFine()
valData:makeCoarse()
valData:makeDiff()

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copying model to gpu')
  model_D:cuda()
  model_G:cuda()
end

-- Training parameters
sgdState_D = {
  learningRate = opt.learningRate,
  momentum = opt.momentum
}

sgdState_G = {
  learningRate = opt.learningRate,
  momentum = opt.momentum
}

-- Get examples to plot
function getSamples(dataset, N)
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  local cond_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local gt_diff = torch.Tensor(N, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local gt = torch.Tensor(N, 3, opt.fineSize, opt.fineSize)

  -- Generate samples
  noise_inputs:uniform(-1, 1)
  for n = 1,N do
    local rand = math.random(dataset:size())
    local sample = dataset[rand]
    cond_inputs[n] = sample[3]:clone()
    gt[n] = sample[4]:clone()
    gt_diff[n] = sample[1]:clone()
  end
  local samples = model_G:forward({noise_inputs, cond_inputs})
  local preds_D = model_D:forward({samples, cond_inputs})

  local to_plot = {}
  for i=1,N do
    local pred = torch.add(cond_inputs[i]:float(), samples[i]:float())
    to_plot[#to_plot+1] = gt[i]:float()
    to_plot[#to_plot+1] = pred
    to_plot[#to_plot+1] = cond_inputs[i]:float()
    to_plot[#to_plot+1] = samples[i]:float()
  end
  return to_plot
end


-- training loop
while true do
  -- train/test
  adversarial.train(trainData, ntrain)
  adversarial.test(valData, nval)

  adversarial.approxParzen(valData, 200, opt.batchSize)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(sgdState_D.learningRate / 1.000004, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(sgdState_G.learningRate / 1.000004, 0.000001)

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    local to_plot = getSamples(valData, 16)
    torch.setdefaulttensortype('torch.FloatTensor')

    trainLogger:style{['% mean class accuracy of D (train set)'] = '-'}
    testLogger:style{['% mean class accuracy of D (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()

    disp.image(to_plot, {win=opt.window, width=700, title=opt.save})
    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
