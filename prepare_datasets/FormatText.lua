require 'torch'
require 'image'
tds = require 'tds'

local cmd = torch.CmdLine()
cmd:option('-rootPath', 'text', 'Path to the text root')
cmd:option('-newPath', 'text_dataset', 'Path to the new data path')
cmd:option('-extension', 'png', 'Extension for the image files')
cmd:option('-height', 30, 'Height of the image preprocess')
cmd:option('-width_pad', 10, 'Height of the image preprocess')

local opt = cmd:parse(arg)

local textTestPath = opt.rootPath .. '/test'
local textTrainPath = opt.rootPath .. '/train'

local function pad_tensor(input, value, paddings)
    local size = input:size()
    assert(#size == #paddings, 'Dimensions mismatch!')
    assert(#size <= 4, 'Can handle tensors with at most 4 dimensions!')
    local new_size = {}
    local region = {}
    for x = 1, #size do
        new_size[x] = size[x] + paddings[x][1] + paddings[x][2]
        region[x * 2 - 1] = paddings[x][1] + 1
        region[x * 2] = paddings[x][1] + size[x]
    end
    local new_tensor = input:clone():resize(unpack(new_size)):fill(value)
    new_tensor:sub(unpack(region)):copy(input)
    return new_tensor
end


local function createDataset(pathToText, newPath)
    sys.execute("mkdir " .. newPath)

    local size = tonumber(sys.execute("find " .. pathToText .. " -type f -name '*'" .. opt.extension .. " | wc -l "))
    local files = io.popen("find -L " .. pathToText .. " -type f -name '*" .. opt.extension .. "'")

    buffer = tds.Vec()
    buffer:resize(size)
 
    counter = 1
    for file in files:lines() do
        buffer[counter] = file
        counter = counter + 1
    end

    for x = 1, size do
        path_ary = sys.split(buffer[x], '/')
        filename = path_ary[#path_ary]
        img_data = image.load(buffer[x], 1, 'byte')
        if (img_data:size():size() == 3) then
            img_data = img_data[1]
        end
        cur_height = img_data:size()[1]
        cur_width = img_data:size()[2]
        new_img_data = torch.Tensor()
        if (cur_height <= opt.height) then
            -- pad the image to get new_image
            top_pad = torch.round((opt.height - cur_height) / 2)
            bottom_pad = opt.height - cur_height - top_pad
        else
            -- scale the image
            new_width = cur_width * opt.height / cur_height
            img_data = image.scale(img_data, new_width, opt.height)
            top_pad = 0
            bottom_pad = 0
        end
        -- pad image
        new_img_data = pad_tensor(img_data, 255, {{top_pad, bottom_pad}, {opt.width_pad, opt.width_pad}})
        image.save(newPath .. filename, new_img_data)

        label = buffer[x]:gsub("png", "txt")
        sys.execute("cp " .. label .. " " .. newPath)
    end

end

sys.execute("rm -rf " .. opt.newPath)
sys.execute("mkdir " .. opt.newPath)
createDataset(textTrainPath, opt.newPath .. '/train/')
createDataset(textTestPath, opt.newPath .. '/test/')