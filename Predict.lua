require 'nn'
require 'audio'
require 'Mapper'
require 'UtilsMultiGPU'
require 'image'
local cmd = torch.CmdLine()
cmd:option('-modelPath', 'deepspeech.t7', 'Path of model to load')
cmd:option('-imagePath', '', 'Path to the input image to predict on')
cmd:option('-dictionaryPath', './equ_dictionary', 'File containing the dictionary to use')
cmd:option('-nGPU', 1)

local opt = cmd:parse(arg)

if opt.nGPU > 0 then
    require 'cunn'
    require 'cudnn'
    require 'BatchBRNNReLU'
end

local model =  loadDataParallel(opt.modelPath, opt.nGPU)
local mapper = Mapper(opt.dictionaryPath)

local img = image.load(opt.imagePath, 1, 'float')
img = img[1]

-- normalize the data
-- local mean = img:mean()
-- std = img:std()
local mean = 0.5
local std = 1
img:add(-mean)
img:div(std)

img = img:view(1, 1, img:size(1), img:size(2))

if opt.nGPU > 0 then
    img = img:cuda()
    model = model:cuda()
end

model:evaluate()
local predictions = model:forward(img)
local tokens = mapper:decodeOutput(predictions[1])
local text = mapper:tokensToText(tokens)

print(text)
