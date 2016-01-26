--[[
--PaddedSequenceBatchTest takes 2 inputs (one is padded with 0's to match
--size of other sequence) and correctly labels them to 2 outputs.
 ]]
require 'nn'
require 'CTCTestCriterion'
require 'optim'
require 'rnn'
local totalNet = nn.Sequential()
local batchSize = 2
torch.manualSeed(12345)
totalNet:add(nn.Sequencer(nn.TemporalConvolution(5,4,1,1)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.Sequencer(nn.TemporalMaxPooling(1,1)))
totalNet:add(nn.Sequencer(nn.TemporalConvolution(4,4,1,1)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.Sequencer(nn.TemporalConvolution(4,4,1,1)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.Sequencer(nn.View(batchSize,-1)))


function findMaxSize(tensors)
    local maxSize = 0
    local allSizes = {}
    for i=1,#tensors do
        local tensorSize = tensors[i]:size(1)
        if(tensorSize > maxSize) then maxSize = tensorSize end
        table.insert(allSizes,tensorSize)
    end
    return allSizes,maxSize
end


local input1 = torch.Tensor({{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},
    {{11,12,13,14,15},{6,7,8,9,10},{0,0,0,0,0}}})
local input2 = torch.Tensor({{11,12,13,14,15},{6,7,8,9,10},{0,0,0,0,0}})
local totalInput = {input1}
local totalTargets = {{2,1,4},{4,1}}
local allSizes,maxSize = findMaxSize(totalInput)
local emptyMax = {}
for i=1,totalInput[1]:size(2) do
table.insert(emptyMax,0)
end
for i=1,#totalInput do
    local input = torch.totable(totalInput[i])
    while(#input < maxSize) do
        table.insert(input,emptyMax)
    end
    totalInput[i] = torch.Tensor(input)
end

print(totalNet:forward(totalInput))

local allSizes,maxSize = findMaxSize(totalInput)
local ctcCriterion = BatchCTCCriterion()
local x, gradParameters = totalNet:getParameters()
function feval(params)
    gradParameters:zero()
    local tensorInput = totalInput
    local targets = totalTargets
    local prediction = totalNet:forward(tensorInput)
    local interleavedPrediction = createInterleavedPrediction(maxSize,prediction)
    local predictionWithSizes = {interleavedPrediction,allSizes}
    local loss = ctcCriterion:forward(predictionWithSizes,targets)
    totalNet:zeroGradParameters()
    local gradOutput = ctcCriterion:backward(predictionWithSizes,targets)
    gradOutput = convertToSequenceData(gradOutput,#totalInput)
    totalNet:backward(tensorInput,gradOutput)
    return loss, gradParameters
end

function convertToSequenceData(gradientOutput, numberOfSequences)
    local gradients = {}
    for i = 1, numberOfSequences do
        table.insert(gradients,{})
    end
    for i = 1, gradientOutput:size(1) do
        if(gradientOutput[i] ~= torch.Tensor(gradientOutput:size(1,1)):zero()) then
            local index = math.fmod(i, numberOfSequences)
            if(index == 0) then index = numberOfSequences end
            table.insert(gradients[index],torch.totable(gradientOutput[i]))
        end
    end
    local returnTensors = {}
    for i = 1, numberOfSequences do
        table.insert(returnTensors,torch.Tensor(gradients[i]))
    end
    return returnTensors
end

function createInterleavedPrediction(maxSize,tensors)
    local columnMajor = {}
    for i = 1,maxSize do
        for index,tensor in ipairs(tensors) do
            table.insert(columnMajor,getTensorValue(tensor,i))
        end
    end
    local resultTensor = torch.Tensor(columnMajor)
    return resultTensor
end

function getTensorValue(tensor,index)
    local tensorValue
    if(tensor:size(1) >= index) then tensorValue = tensor[index] end
    if(tensorValue == nil) then
        local emptyTensor = torch.totable(tensor[1]:zero())
        return emptyTensor
    else
        local tableTensorValue = torch.totable(tensorValue)
        return tableTensorValue end
end

local sgd_params = {
    learningRate = 0.1,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0.9
}

local currentLoss = 300
local iteration = 1
while iteration < 200 do
    currentLoss = 0
    iteration = iteration + 1
    local _,fs = optim.sgd(feval,x,sgd_params)
    currentLoss = currentLoss + fs[1]
    print("Loss: ",currentLoss, " Iteration: ", iteration)
end
print(totalNet:forward(totalInput)[1])
print(totalNet:forward(totalInput)[2])