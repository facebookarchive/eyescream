require 'torch'
require 'cunn'
require 'nngraph'
require 'optim'
require 'image'
require 'datasets.coarse_to_fine_cifar10'
require 'pl'
require 'paths'
image_utils = require 'utils.image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'train.double_conditional_adversarial'
require 'layers.SpatialConvolutionUpsample'


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 2)           save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.02)        learning rate
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0)           momentum
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
opt.condDim1 = 10
opt.condDim2 = {3, opt.fineSize, opt.fineSize}
cifar_classes =  {'airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ----------------------------------------------------------------------
  -- define D network to train
  local nplanes = opt.hidden_D
  x_d = nn.Identity()()
  x_c1 = nn.Identity()()
  x_c2 = nn.Identity()()
  d1 = nn.CAddTable()({x_d, x_c2})
  c1 = nn.Linear(opt.condDim1, opt.condDim2[2]*opt.condDim2[3])(x_c1)
  c2 = nn.Reshape(1, opt.condDim2[2], opt.condDim2[3])(nn.ReLU()(c1))
  d2 = nn.JoinTable(2, 2)({d1, c2})
  d3 = nn.SpatialConvolution(3+1, nplanes, 5, 5)(d2)
  d4 = nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2)(nn.ReLU()(d3))
  local sz =math.floor( ( (opt.fineSize - 5 + 1) - 5) / 2 + 1)
  d5 = nn.Reshape(nplanes*sz*sz)(d4)
  d6 = nn.Linear(nplanes*sz*sz, 1)(nn.Dropout()(nn.ReLU()(d5)))
  d7 = nn.Sigmoid()(d6)
  model_D = nn.gModule({x_d, x_c1, x_c2}, {d7})

  ----------------------------------------------------------------------
  -- define G network to train
  local nplanes = opt.hidden_G
  x_n = nn.Identity()() -- noise (shaped as coarse map)
  g_c1 = nn.Identity()() -- class vector
  g_c2 = nn.Identity()() -- coarse map
  class1 = nn.Linear(opt.condDim1, opt.condDim2[2]*opt.condDim2[3])(g_c1)
  class2 = nn.Reshape(1, opt.condDim2[2], opt.condDim2[3])(nn.ReLU()(class1)) --convert class vector into map
  g1 = nn.JoinTable(2, 2)({x_n, class2, g_c2}) -- combine maps into 5 channels
  g2 = nn.SpatialConvolutionUpsample(5, nplanes, 7, 7, 1)(g1)
  g3 = nn.SpatialConvolutionUpsample(nplanes, nplanes, 7, 7, 1)(nn.ReLU()(g2))
  g4 = nn.SpatialConvolutionUpsample(nplanes, 3, 5, 5, 1)(nn.ReLU()(g3))
  model_G = nn.gModule({x_n, g_c1, g_c2}, {g4})

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

local nparams = 0
for i=1,#model_D.forwardnodes do
  if model_D.forwardnodes[i].data.module ~= nil and model_D.forwardnodes[i].data.module.weight ~= nil then
    nparams = nparams + model_D.forwardnodes[i].data.module.weight:nElement()
  end
end
print('\nNumber of free parameters in D: ' .. nparams)

local nparams = 0
for i=1,#model_G.forwardnodes do
  if model_G.forwardnodes[i].data.module ~= nil and model_G.forwardnodes[i].data.module.weight ~= nil then
    nparams = nparams + model_G.forwardnodes[i].data.module.weight:nElement()
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
function getSamples(dataset, N, perclass)
  local N = N or 8
  local perclass = perclass or 10
  local noise_inputs = torch.Tensor(N, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  local cond_inputs1 = torch.Tensor(N, opt.condDim1)
  local cond_inputs2 = torch.Tensor(N, opt.condDim2[1], opt.condDim2[2], opt.condDim2[3])
  local gt_diff = torch.Tensor(N, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local gt = torch.Tensor(N, 3, opt.fineSize, opt.fineSize)

  -- Generate samples
  noise_inputs:uniform(-1, 1)
  local class = 1
  local classes = {}
  for n = 1,N do
    classes[n] = cifar_classes[class]
    local rand
    local sample
    while true do
      rand = math.random(dataset:size())
      sample = dataset[rand]
      local max, ind = torch.max(sample[2], 1)
      if ind[1] == class then
        break
      end
    end
    cond_inputs1[n] = sample[2]:clone()
    cond_inputs2[n] = sample[3]:clone()
    gt[n] = sample[4]:clone()
    gt_diff[n] = sample[1]:clone()
    if n % perclass == 0 then class = class + 1 end
    if class > #cifar_classes then class = 1 end
  end
  local samples = model_G:forward({noise_inputs, cond_inputs1, cond_inputs2})
  local preds_D = model_D:forward({samples, cond_inputs1, cond_inputs2})

  local to_plot = {}
  for i=1,N do
    local pred = torch.add(cond_inputs2[i]:float(), samples[i]:float())
    to_plot[#to_plot+1] = gt[i]:float()
    to_plot[#to_plot+1] = pred
    to_plot[#to_plot+1] = cond_inputs2[i]:float()
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
  sgdState_D.learningRate = math.max(sgdState_D.learningRate / 1.00004, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(sgdState_G.learningRate / 1.00004, 0.000001)

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    local to_plot = getSamples(valData, 16, 2)
    torch.setdefaulttensortype('torch.FloatTensor')

    trainLogger:style{['% mean class accuracy of D (train set)'] = '-'}
    testLogger:style{['% mean class accuracy of D (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()

    disp.image(to_plot, {win=opt.window, width=600, title=opt.save})
    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
