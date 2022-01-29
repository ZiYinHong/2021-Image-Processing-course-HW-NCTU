man = imread('astronaut-interference.tif'); 
manf=fftshift(fft2(man));

figure,imshow(man) 
figure,fftshow(manf)