require 'cudnn'
require 'cunn'
paths.dofile('cudnnSpatialConvolutionUpsample.lua')
paths.dofile('SpatialConvolutionUpsample.lua')

local cudnntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local precision_jac = 1e-3
local nloop = 1
local times = {}
local mytester

function cudnntest.SpatialConvolution_forward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,5) * 2 - 1
   local kj = ki
   local scale = math.random(1,2)
   local outi = math.random(1,32) * 2
   local outj = outi
   local ini = outi / scale
   local inj = outj / scale
   local input = torch.randn(bs,from,inj,ini):cuda()
   local sconv = nn.SpatialConvolutionUpsample(from,to,ki,kj,scale):cuda()
   local groundtruth = sconv:forward(input)
   cutorch.synchronize()
   local gconv = cudnn.SpatialConvolutionUpsample(from,to,ki,kj,scale):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local rescuda = gconv:forward(input)
   cutorch.synchronize()
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cudnntest.SpatialConvolution_backward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32) * 2
   local to = math.random(1,64) * 2
   local ki = math.random(1,5) * 2 - 1
   local kj = ki
   local scale = math.random(1,2)
   local outi = math.random(ki,32) * 2
   local outj = outi
   local ini = outi / scale
   local inj = outj / scale
   print(bs, from, to, inj, ini, outj, outi, scale, ki, kj)

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,to,outj,outi):cuda()
   local sconv = nn.SpatialConvolutionUpsample(from,to,ki,kj,scale):cuda()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cudnn.SpatialConvolutionUpsample(from,to,ki,kj,scale):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local error = rescuda:float() - groundgrad:float()
   local werror = weightcuda:float() - groundweight:float()
   local berror = biascuda:float() - groundbias:float()

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(cudnntest)

mytester:run()

os.execute('rm -f modelTemp.t7')
