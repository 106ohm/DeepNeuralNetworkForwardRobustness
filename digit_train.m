% Johannes Langelaar (2020). MNIST neural network training and testing (https://www.mathworks.com/matlabcentral/fileexchange/73010-mnist-neural-network-training-and-testing), MATLAB Central File Exchange. Retrieved May 25, 2020.

data = load('mnist_train.csv');

af = 'ReLU';

labels = data(:,1); % (supervisedly) correct digits, from 0 to 9
y = zeros(10,60000); %Correct outputs vector
for i = 1:60000
    y(labels(i)+1,i) = 1;% if the correct digit is "i" then the correct 
                         % out layer is e_{i+1}
end

images = data(:,2:785); % the 28*28=784 pixel image is stored by row
images = images/255; % gray values from 0 to 255 are scaled

images = images'; %Input vectors

hn1 = 80; %Number of neurons in the first hidden layer
hn2 = 60; %Number of neurons in the second hidden layer

%Initializing weights and biases
w12 = randn(hn1,784)*sqrt(2/784);
w23 = randn(hn2,hn1)*sqrt(2/hn1);
w34 = randn(10,hn2)*sqrt(2/hn2);
b12 = randn(hn1,1);
b23 = randn(hn2,1);
b34 = randn(10,1);

%learning rate
eta = 0.0058;

%Initializing errors and gradients
error4 = zeros(10,1);
error3 = zeros(hn2,1);
error2 = zeros(hn1,1);
errortot4 = zeros(10,1);
errortot3 = zeros(hn2,1);
errortot2 = zeros(hn1,1);
grad4 = zeros(10,1);
grad3 = zeros(hn2,1);
grad2 = zeros(hn1,1);

epochs = 50;

m = 10; %Minibatch size

for k = 1:epochs %Outer epoch loop
    
    batches = 1;
    
    for j = 1:60000/m
        error4 = zeros(10,1);
        error3 = zeros(hn2,1);
        error2 = zeros(hn1,1);
        errortot4 = zeros(10,1);
        errortot3 = zeros(hn2,1);
        errortot2 = zeros(hn1,1);
        grad4 = zeros(10,1);
        grad3 = zeros(hn2,1);
        grad2 = zeros(hn1,1);
    for i = batches:batches+m-1 %Loop over each minibatch
    
    %Feed forward
    a1 = images(:,i);
    z2 = w12*a1 + b12;
    a2 = actfun(z2,af);
    z3 = w23*a2 + b23;
    a3 = actfun(z3,af);
    z4 = w34*a3 + b34;
    a4 = actfun(z4,af); %Output vector
    
    %backpropagation
    error4 = (a4-y(:,i)).*actfunprime(z4,af);
    error3 = (w34'*error4).*actfunprime(z3,af);
    error2 = (w23'*error3).*actfunprime(z2,af);
    
    errortot4 = errortot4 + error4;
    errortot3 = errortot3 + error3;
    errortot2 = errortot2 + error2;
    grad4 = grad4 + error4*a3';
    grad3 = grad3 + error3*a2';
    grad2 = grad2 + error2*a1';

    end
    
    %Gradient descent
    w34 = w34 - eta/m*grad4;
    w23 = w23 - eta/m*grad3;
    w12 = w12 - eta/m*grad2;
    b34 = b34 - eta/m*errortot4;
    b23 = b23 - eta/m*errortot3;
    b12 = b12 - eta/m*errortot2;
    
    batches = batches + m;
    
    end
    fprintf('Epochs:');
    disp(k) %Track number of epochs
    [images,y] = shuffle(images,y); %Shuffles order of the images for next epoch
end

disp('Training done!')
%Saves the parameters
save(strcat(af,'_wfour.mat'),'w34');
save(strcat(af,'_wthree.mat'),'w23');
save(strcat(af,'_wtwo.mat'),'w12');
save(strcat(af,'_bfour.mat'),'b34');
save(strcat(af,'_bthree.mat'),'b23');
save(strcat(af,'_btwo.mat'),'b12');