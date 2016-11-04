-- Expects data in the format of <root><train/test><datasetname><filename.wav/filename.txt>
-- Creates an LMDB of everything in these folders into a train and test set.

require 'lfs'
require 'xlua'
require 'lmdb'
require 'torch'
require 'parallel'
require 'image'

local tds = require 'tds'

local cmd = torch.CmdLine()
cmd:option('-rootPath', 'prepare_datasets/equ_dataset', 'Path to the data')
cmd:option('-lmdbPath', 'prepare_datasets/equ_lmdb', 'Path to save LMDBs to')
cmd:option('-processes', 2, 'Number of processes used to create LMDB')
cmd:option('-imageExtension', 'png', 'The extension of the image files (png/jpg)')

local opt = cmd:parse(arg)
local dataPath = opt.rootPath
local lmdbPath = opt.lmdbPath
local extension = '.' .. opt.imageExtension
parallel.nfork(opt.processes)

local function startWriter(path, name)
    local db = lmdb.env {
        Path = path,
        Name = name
    }
    db:open()
    local txn = db:txn()
    return db, txn
end

local function closeWriter(db, txn)
    txn:commit()
    db:close()
end

local function createLMDB(dataPath, lmdbPath, id)
    local vecs = tds.Vec()
    local sortIdsPath = 'sort_ids_'.. id .. '.t7' -- in case of crash, sorted ids are saved

    local function file_exists(name)
        local f = io.open(name, "r")
        if f ~= nil then io.close(f) return true else return false end
    end

    if not file_exists(sortIdsPath) then
        local size = tonumber(sys.execute("find " .. dataPath .. " -type f -name '*'" .. extension .. " | wc -l "))
        vecs:resize(size)

        local files = io.popen("find -L " .. dataPath .. " -type f -name '*" .. extension .. "'")
        local counter = 1
        print("Retrieving sizes for sorting...")
        local buffer = tds.Vec()
        buffer:resize(size)

        for file in files:lines() do
            buffer[counter] = file
            counter = counter + 1
        end

        local function getSize(opts)
            local imageFilePath = opts.file
            local labelFilePath = opts.file:gsub(opts.extension, ".txt")
            local opt = opts.opt
            local imageFile = image.load(imageFilePath, 1)
            local length = imageFile:size()[2]
            return { imageFilePath, labelFilePath, length }
        end

        for x = 1, opt.processes do
            local opts = { extension = extension, file = buffer[x], opt = opt }
            parallel.children[x]:send({ opts, getSize })
        end

        local processCounter = 1
        for x = 1, size do
            local result = parallel.children[processCounter]:receive()
            vecs[x] = tds.Vec(unpack(result))
            xlua.progress(x, size)
            if x % 1000 == 0 then collectgarbage() end
            -- send next index to retrieve
            if x + opt.processes <= size then
                local opts = { extension = extension, file = buffer[x + opt.processes], opt = opt }
                parallel.children[processCounter]:send({ opts, getSize })
            end
            if processCounter == opt.processes then
                processCounter = 1
            else
                processCounter = processCounter + 1
            end
        end
        print("Sorting...")
        -- sort the files by length
        local function comp(a, b) return a[3] < b[3] end

        vecs:sort(comp)
        torch.save(sortIdsPath, vecs)
    else
        vecs = torch.load(sortIdsPath)
    end
    local size = #vecs

    print("Creating LMDB dataset to: " .. lmdbPath)
    -- start writing
    local dbImage, readerImage = startWriter(lmdbPath .. '/images', 'images')
    local dbLabel, readerLabel = startWriter(lmdbPath .. '/labels', 'labels')

    local function getData(opts)
        local opt = opts.opt
        local imageData = image.load(opts.imageFilePath, 1, 'float')

        -- put into lmdb

        -- normalize the data
        local mean = imageData:mean()
        local std = imageData:std()
        imageData:add(-mean)
        imageData:div(std)

        local label
        for line in io.lines(opts.labelFilePath) do
            label = line
        end
        return { imageData, label }
    end

    for x = 1, opt.processes do
        local vec = vecs[x]
        local opts = { imageFilePath = vec[1], labelFilePath = vec[2], opt = opt }
        parallel.children[x]:send({ opts, getData })
    end

    local processCounter = 1
    for x = 1, size do
        local result = parallel.children[processCounter]:receive()
        local imageData, label = unpack(result)

        readerImage:put(x, imageData)
        readerLabel:put(x, label)

        -- commit buffer
        if x % 500 == 0 then
            readerImage:commit(); readerImage = dbImage:txn()
            readerLabel:commit(); readerLabel = dbLabel:txn()
            collectgarbage()
        end

        if x + opt.processes <= size then
            local vec = vecs[x + opt.processes]
            local opts = { imageFilePath = vec[1], labelFilePath = vec[2], opt = opt }
            parallel.children[processCounter]:send({ opts, getData })
        end
        if processCounter == opt.processes then
            processCounter = 1
        else
            processCounter = processCounter + 1
        end
        xlua.progress(x, size)
    end

    closeWriter(dbImage, readerImage)
    closeWriter(dbLabel, readerLabel)
end

function parent()
    local function looper()
        require 'torch'
        require 'image'
        while true do
            local object = parallel.parent:receive()
            local opts, code = unpack(object)
            local result = code(opts)
            parallel.parent:send(result)
            collectgarbage()
        end
    end

    parallel.children:exec(looper)

    createLMDB(dataPath .. '/train', lmdbPath .. '/train', 'equ.train')
    createLMDB(dataPath .. '/test', lmdbPath .. '/test', 'equ.test')
    parallel.close()
end

sys.execute("rm -rf " .. opt.lmdbPath)
sys.execute("mkdir " .. opt.lmdbPath)
local ok, err = pcall(parent)
if not ok then
    print(err)
    parallel.close()
end