test = load('mnist_test.csv');
labels = test(:,1);

%% parameter
imag = 37;

%% retrive image
image = test(imag,2:785);
image = image/255;

image = image';

%% define random matrices
w2=rand(784);
b2=rand(784,1);
w3=rand(784);
b3=rand(784,1);
w4=rand(784);
b4=rand(784,1);

%% invert the matrices
iw2=inv(w2);
iw3=inv(w3);
iw4=inv(w4);

%% apply forward the Neural Network
af = 'LeakyReLU';
% af = 'Linear';
out1 = actfun(w2*image+b2,af);
out2 = actfun(w3*out1+b3,af);
out = w4*out2+b4;

%% apply backward the Neural Network
af = 'LeakyReLU_inverse';
% af = 'Linear';
out2i = iw4*(out-b4);
out1i = actfun(iw3*(out2i-b3),af);
imagei = actfun(iw2*(out1i-b2),af);

diff = image-imagei;

max(abs(diff))