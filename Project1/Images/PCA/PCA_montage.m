dirOutput = dir('*.png');
fileNames = string({dirOutput.name});

% montage(fileNames,'ThumbnailSize',[]);%,'Size',[4 4]

for i=1:16
    I = imread(fileNames(i));
    subplot(4,4,i)
    imshow(I);
end
