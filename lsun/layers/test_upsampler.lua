require 'nn'
paths.dofile('../layers/SpatialConvolutionUpsample.lua')
local mytester = torch.Tester()
local jac
local precision = 1e-5

nntest = {}
function nntest.SpatialConvolutionUpsample()
   local from = math.random(1,5)
   local to = math.random(1,5)
   local ki = 3
   local kj = 3
   local outi = 16
   local outj = 16
   local factor = 2
   local ini = outi / factor
   local inj = outj / factor
   local module = nn.SpatialConvolutionUpsample(from, to, ki, kj, factor)
   local input = torch.Tensor(from, inj, ini):zero()

   -- stochastic

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- batch

   local batch = math.random(2,5)
   module = nn.SpatialConvolutionUpsample(from, to, ki, kj, factor)
   input = torch.Tensor(batch,from,inj,ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'batch error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'batch error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'batch error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

mytester:add(nntest)

require 'nn'
jac = nn.Jacobian
mytester:run()
