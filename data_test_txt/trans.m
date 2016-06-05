fileFolder = fullfile('/Users/makai/Documents/projects/ImageProcessing/editdistance/data_txt/data_test');
dirOutput = dir(fullfile(fileFolder, '*.txt'));
fileNames = {dirOutput.name};

for i = 1 : length(fileNames)
    cor2img(fileNames{i});
end