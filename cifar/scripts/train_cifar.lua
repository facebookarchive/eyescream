require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'datasets.scaled_cifar10'
require 'pl'
require 'paths'
image_utils = require 'utils.image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'train.adversarial'


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 10)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.02)        learning rate
  -b,--batchSize     (default 100)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default -1)          gpu to run on (default cpu)
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --hidden_G         (default 8000)        number of units in hidden layers of G
  --hidden_D         (default 1600)        number of units in hidden layers of D
  --scale            (default 32)          scale of images to train on
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


classes = {'0','1'}
opt.geometry = {3, opt.scale, opt.scale}

function setWeights(weights, std)
  weights:randn(weights:size())
  weights:mul(std)
end

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ----------------------------------------------------------------------
  -- define D network to train
  local numhid = opt.hidden_D
  model_D = nn.Sequential()
  model_D:add(nn.Reshape(input_sz))
  model_D:add(nn.Linear(input_sz, numhid))
  model_D:add(nn.ReLU())
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(numhid, numhid))
  model_D:add(nn.ReLU())
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(numhid,1))
  model_D:add(nn.Sigmoid())

  -- Init weights
  setWeights(model_D.modules[2].weight, 0.005)
  setWeights(model_D.modules[5].weight, 0.005)
  setWeights(model_D.modules[8].weight, 0.005)
  setWeights(model_D.modules[2].bias, 0)
  setWeights(model_D.modules[5].bias, 0)
  setWeights(model_D.modules[8].bias, 0)

  ----------------------------------------------------------------------
  -- define G network to train
  local numhid = opt.hidden_G
  model_G = nn.Sequential()
  model_G:add(nn.Linear(opt.noiseDim, numhid))
  model_G:add(nn.ReLU())
  model_G:add(nn.Linear(numhid, numhid))
  model_G:add(nn.Sigmoid())
  model_G:add(nn.Linear(numhid, input_sz))
  model_G:add(nn.Reshape(opt.geometry[1], opt.geometry[2], opt.geometry[3]))

  -- Init weights
  setWeights(model_G.modules[1].weight, 0.05)
  setWeights(model_G.modules[3].weight, 0.05)
  setWeights(model_G.modules[5].weight, 0.05)
  setWeights(model_G.modules[1].bias, 0)
  setWeights(model_G.modules[3].bias, 0)
  setWeights(model_G.modules[5].bias, 0)
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
print('Generator network:')
print(model_G)


----------------------------------------------------------------------
-- get/create dataset
--
ntrain = 45000
nval = 5000

cifar.setScale(opt.scale)

-- create training set and normalize
trainData = cifar.loadTrainSet(1, ntrain)
mean, std = image_utils.normalize(trainData.data)
trainData:scaleData()

-- create validation set and normalize
valData = cifar.loadTrainSet(ntrain+1, ntrain+nval)
image_utils.normalize(valData.data, mean, std)
valData:scaleData()


-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
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
  local numperclass = numperclass or 10
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.noiseDim)

  -- Generate samples
  noise_inputs:uniform(-1, 1)
  local samples = model_G:forward(noise_inputs)

  local to_plot = {}
  for i=1,N do
    to_plot[#to_plot+1] = samples[i]:float()
  end

  return to_plot
end


-- training loop
while true do
  -- train/test
  adversarial.train(trainData)
  adversarial.test(valData)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(sgdState_D.learningRate / 1.000004, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(sgdState_G.learningRate / 1.000004, 0.000001)

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    local to_plot = getSamples(valData, 100)
    torch.setdefaulttensortype('torch.FloatTensor')

    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
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
