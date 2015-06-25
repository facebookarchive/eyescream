paths.dofile('modelGen.lua')

opt = opt or {}
opt.geometry = opt.geometry or {3,16,16}
print('\nGenerator')
model_G, desc_G = generateModelG(2,5,64,1024,3,11, 'mixed', 0, 4, 2)
print(desc_G)
print('\nDiscriminator')
model_D, desc_D = generateModelD(2,6,64,1024,3,11, 'mixed', 0, 4, 2)
print(desc_D)
