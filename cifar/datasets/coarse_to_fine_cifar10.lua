--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'torch'
require 'paths'
require 'image'
image_utils = require 'utils.image'

cifar = {}

cifar.path_dataset = '/your/path/to/cifar10/cifar-10-batches-t7/'

cifar.coarseSize = 16
cifar.fineSize = 32

function cifar.init(fineSize, coarseSize)
  cifar.fineSize = fineSize
  cifar.coarseSize = coarseSize
end

function cifar.loadTrainSet(start, stop, augment, crop)
   return cifar.loadDataset(true, start, stop, augment, crop)
end

function cifar.loadTestSet(crop)
   return cifar.loadDataset(false, nil, nil, nil, crop)
end

function cifar.loadDataset(isTrain, start, stop, augment, crop)
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
    data = subset.data:t():reshape(10000, 3, 32, 32):type('torch.FloatTensor')
    labels = subset.labels:t():type(defaultType)
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
  dataset.data = data -- on cpu
  dataset.labels = labels

  dataset.coarseData = torch.FloatTensor(N, 3, cifar.fineSize, cifar.fineSize)
  dataset.fineData = torch.FloatTensor(N, 3, cifar.fineSize, cifar.fineSize)
  dataset.diffData = torch.FloatTensor(N, 3, cifar.fineSize, cifar.fineSize)

  -- Coarse data
  function dataset:makeCoarse()
    for i = 1,N do
      local tmp = image.scale(self.data[i], cifar.coarseSize, cifar.coarseSize)
      self.coarseData[i] = image.scale(tmp, cifar.fineSize, cifar.fineSize)
    end
  end

  -- Fine data
  function dataset:makeFine()
    for i = 1,N do
      self.fineData[i] = image.scale(self.data[i], cifar.fineSize, cifar.fineSize)
    end
  end

  -- Diff (coarse - fine)
  function dataset:makeDiff()
    for i=1,N do
      self.diffData[i] = torch.add(self.fineData[i], -1, self.coarseData[i])
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
     local diff = self.diffData[index]
     local cond = self.coarseData[index]
     local fine = self.fineData[index]
     labelvector:zero()
     labelvector[self.labels[index]] = 1
     local example = {diff, labelvector, cond, fine}
     return example
end})

  return dataset
end
