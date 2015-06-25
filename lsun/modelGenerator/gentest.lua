--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

paths.dofile('modelGen.lua')

opt = opt or {}
opt.geometry = opt.geometry or {3,16,16}
print('\nGenerator')
model_G, desc_G = generateModelG(2,5,64,1024,3,11, 'mixed', 0, 4, 2)
print(desc_G)
print('\nDiscriminator')
model_D, desc_D = generateModelD(2,6,64,1024,3,11, 'mixed', 0, 4, 2)
print(desc_D)
