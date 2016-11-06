require 'UtilsMultiGPU'

local function RNNModule(inputDim, hiddenDim, opt)
    if opt.nGPU > 0 then
        if opt.rnn.LSTM then
            local blstm = nn.Sequential()
            blstm:add(cudnn.BLSTM(inputDim, hiddenDim, 1))
            blstm:add(nn.View(-1, 2, hiddenDim):setNumInputDims(2)) -- have to sum activations
            blstm:add(nn.Sum(3))
            return blstm
        else
            require 'BatchBRNNReLU'
            return cudnn.BatchBRNNReLU(inputDim, hiddenDim)
        end
    else
        require 'rnn'
        return nn.SeqBRNN(inputDim, hiddenDim)
    end
end

-- Creates the covnet+rnn structure.
local function deepSpeech(opt)
    local conv = nn.Sequential()
    assert(#opt.cnn.channels == #opt.cnn.kernel_heights and #opt.cnn.kernel_heights == #opt.cnn.kernel_widths,
        'Arrays of channels, kernel_heights and kernel_widths must have the same length.')
    -- insert the input channel size: 1
    table.insert(opt.cnn.channels, 1, 1)
    local cnn_output_height = opt.input_height
    local cnn_output_channel
    for x = 1, #opt.cnn.kernel_heights do
        -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
        conv:add(nn.SpatialConvolution(opt.cnn.channels[x],
                                       opt.cnn.channels[x + 1],
                                       opt.cnn.kernel_widths[x],
                                       opt.cnn.kernel_heights[x],
                                       1,
                                       1))
        if opt.cnn.with_bn then
            conv:add(nn.SpatialBatchNormalization(opt.cnn.channels[x + 1]))
        end
        conv:add(nn.Clamp(0, 20))
        cnn_output_height = cnn_output_height - opt.cnn.kernel_heights[x] + 1
        cnn_output_channel = opt.cnn.channels[x + 1]
    end
    local rnnInputsize = cnn_output_channel * cnn_output_height -- based on the above convolutions and 16khz audio.
    local rnnHiddenSize = opt.rnn.hiddenSize -- size of rnn hidden layers
    local nbOfHiddenLayers = opt.rnn.nbOfHiddenLayers

    conv:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x features x seqLength
    conv:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    local rnns = nn.Sequential()
    local rnnModule = RNNModule(rnnInputsize, rnnHiddenSize, opt)
    rnns:add(rnnModule:clone())
    rnnModule = RNNModule(rnnHiddenSize, rnnHiddenSize, opt)

    for i = 1, nbOfHiddenLayers - 1 do
        rnns:add(nn.Bottle(nn.BatchNormalization(rnnHiddenSize), 2))
        rnns:add(rnnModule:clone())
    end

    local fullyConnected = nn.Sequential()
    fullyConnected:add(nn.BatchNormalization(rnnHiddenSize))
    fullyConnected:add(nn.Linear(rnnHiddenSize, opt.label_size))

    local model = nn.Sequential()
    model:add(conv)
    model:add(rnns)
    model:add(nn.Bottle(fullyConnected, 2))
    model:add(nn.Transpose({1, 2})) -- batch x seqLength x features
    model = makeDataParallel(model, opt.nGPU)
    return model
end

-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes, opt)
    for x = 1, #opt.cnn.kernel_heights do
        sizes = sizes - opt.cnn.kernel_heights[x] + 1
    end
    -- sizes = torch.floor((sizes - 9) + 1) -- conv1
    -- sizes = torch.floor((sizes - 9) + 1) -- conv2
    return sizes
end

return { deepSpeech, calculateInputSizes }