require 'torch'
require 'image'

local img = {}

function img.normalize(data, mean_, std_)
  local mean = mean_ or data:mean(1)
  local std = std_ or data:std(1, true)
  local eps = 1e-7
  for i=1,data:size(1) do
    data[i]:add(-1, mean)
    data[i]:cdiv(std + eps)
  end
  return mean, std
end

function img.normalizeGlobal(data, mean_, std_)
  local std = std_ or data:std()
  local mean = mean_ or data:mean()
  data:add(-mean)
  data:mul(1/std)
  return mean, std
end

function img.contrastNormalize(data, new_min, new_max)
  local old_max = data:max(1)
  local old_min = data:min(1)
  local eps = 1e-7
  for i=1,data:size(1) do
    data[i]:add(-1, old_min)
    data[i]:mul(new_max - new_min)
    data[i]:cdiv(old_max - old_min + eps)
    data[i]:add(new_min)
  end
end

function img.flip(data, labels)
  local n = data:size(1)
  local N = n*2
  local new_data = torch.Tensor(N, data:size(2), data:size(3), data:size(4)):typeAs(data)
  local new_labels = torch.Tensor(N)
  new_data[{{1,n}}] = data
  new_labels[{{1,n}}] = labels:clone()
  new_labels[{{n+1,N}}] = labels:clone()
  for i = n+1,N do
    new_data[i] = image.hflip(data[i-n])
  end
  local rp = torch.LongTensor(N)
  rp:randperm(N)
  return new_data:index(1, rp), new_labels:index(1, rp)
end

function img.translate(data, w, labels)
  local n = data:size(1)
  local N = n*5
  local ow = data:size(3)
  local new_data = torch.Tensor(N, data:size(2), w, w):typeAs(data)
  local new_labels = torch.Tensor(N)
  local d = ow - w + 1
  local m1 = (ow - w) / 2 + 1
  local m2 = ow - ((ow - w) / 2)
  local x1 = {1, d, 1, d, m1} 
  local x2 = {w, ow, w, ow, m2}
  local y1 = {1, 1, d, d, m1}
  local y2 = {w, w, ow, ow, m2}
  local k = 1
  for i = 1,n do
    for j = 1,5 do
      new_data[k] = data[{ i, {}, {y1[j], y2[j]}, {x1[j], x2[j]} }]:clone()
      new_labels[k] = labels[i]
      k = k + 1
    end
  end
  local rp = torch.LongTensor(N)
  rp:randperm(N)
  return new_data:index(1, rp), new_labels:index(1, rp)
end

return img
