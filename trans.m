function trans()
    fileFolder = fullfile('/Users/makai/Desktop/aa/data_test_txt');
    dirOutput = dir(fullfile(fileFolder, '*.txt'));
    fileNames = {dirOutput.name};

    for i = 1 : length(fileNames)
        cor2img(['data_test_txt/', fileNames{i}]);
    end
    exit();