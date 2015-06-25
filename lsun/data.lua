local Threads = require 'threads'

local donkey_file
if opt.dataset == 'imagenet' then
   donkey_file = 'donkey_imagenet.lua'
elseif opt.dataset == 'lsun' then
   donkey_file = 'donkey_lsun.lua'
   -- lmdb complains beyond 6 donkeys. wtf.
   if opt.nDonkeys > 6 and opt.forceDonkeys == 0 then opt.nDonkeys = 6 end
else
   error('Unknown dataset: ', opt.dataset)
end

do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      donkeys = Threads(
         opt.nDonkeys,
         function()
            require 'torch'
         end,
         function(idx)
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            local seed = opt.manualSeed + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            paths.dofile(donkey_file)
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile(donkey_file)
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

os.execute('mkdir -p '.. opt.save)

nTest = 0
donkeys:addjob(function() return testLoader:size() end, function(c) nTest = c end)
donkeys:synchronize()
assert(nTest > 0, "Failed to get nTest")
print('nTest: ', nTest)


function sanitize(net)
   local list = net:listModules()
   for _,val in ipairs(list) do
      for name,field in pairs(val) do
         if torch.type(field) == 'cdata' then val[name] = nil end
         if name == 'homeGradBuffers' then val[name] = nil end
         if name == 'input_gpu' then val['input_gpu'] = {} end
         if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
         if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
         if (name == 'output' or name == 'gradInput') then
            if torch.isTensor(val[name]) then
               val[name] = field.new()
            end
         end
         if  name == 'buffer' or name == 'buffer2' or name == 'normalized'
         or name == 'centered' or name == 'addBuffer' then
            val[name] = nil
         end
      end
   end
   return net
end

function merge_table(t1, t2)
   local t = {}
   for k,v in pairs(t2) do
      t[k] = v
   end
   for k,v in pairs(t1) do
      t[k] = v
   end
   return t
end
