--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'torch'
require 'image'
require 'paths'

cifar = {}

cifar.path_dataset = '/your/path/to/cifar10/cifar-10-batches-t7/'
cifar.scale = 32

function cifar.setScale(scale)
  cifar.scale = scale
end

function cifar.loadTrainSet(start, stop)
  return cifar.loadDataset(true, start, stop)
end

function cifar.loadTestSet()
  return cifar.loadDataset(false)
end

function cifar.loadDataset(isTrain, start, stop)
  local data
  local labels
  local defaultType = torch.getdefaulttensortype()
  if isTrain then -- load train data
    data = torch.FloatTensor(50000, 3, 32, 32)
    labels = torch.FloatTensor(50000)
    for i = 0,4 do
      local subset = torch.load(cifar.path_dataset .. 'data_batch_' .. (i+1) .. '.t7', 'ascii')
      data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():reshape(10000, 3, 32, 32)
      labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end
  else -- load test data
    subset = torch.load(cifar.path_dataset .. 'test_batch.t7', 'ascii')
    data = subset.data:t():reshape(10000, 3, 32, 32)
    labels = subset.labels:reshape(10000)
  end

  local start = start or 1
  local stop = stop or data:size(1)

  -- select chunk
  data = data[{ {start, stop} }]
  labels = labels[{ {start, stop} }]
  labels:add(1) -- becasue indexing is 1-based
  local N = stop - start + 1
  print('<cifar10> loaded ' .. N .. ' examples')

  local dataset = {}
  dataset.data = data
  dataset.labels = labels
  dataset.scaled = torch.Tensor(N, 3, cifar.scale, cifar.scale)

  function dataset:scaleData()
    for n = 1,N do
      dataset.scaled[n] = image.scale(dataset.data[n], cifar.scale, cifar.scale)
    end
  end


  function dataset:size()
    return N
  end

  function dataset:numClasses()
    return 10
  end

  local labelvector = torch.zeros(10)

  setmetatable(dataset, {__index = function(self, index)
    local input = self.scaled[index]
    local class = self.labels[index]
    local label = labelvector:zero()
    label[class] = 1
    local example = {input, class, label}
    return example
  end})

  return dataset
end
