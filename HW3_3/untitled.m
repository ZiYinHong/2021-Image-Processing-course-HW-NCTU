I = imread("Body.tif");
imshow(I);

%%
g=fspecial('gaussian',[255 255], 64);

blur_INT = imfilter(I, g, 'same');
blur = imfilter(im2double(I), im2double(g), 'same');
imshow(blur);

%%
output = im2double(I)./blur;
imshow(output, [0 1]);

%%
blur2 = filter2(g, I, "same");
imshow(blur2);

%%
output2 = im2double(I)./blur2;
imshow(output2, [0 1]);