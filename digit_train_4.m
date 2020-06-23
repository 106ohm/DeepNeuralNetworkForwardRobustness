% Johannes Langelaar (2020). MNIST neural network training and testing (https://www.mathworks.com/matlabcentral/fileexchange/73010-mnist-neural-network-training-and-testing), MATLAB Central File Exchange. Retrieved May 25, 2020.

data = load('mnist_train.csv');

af = 'LeakyReLU';

labels = data(:,1); % (supervisedly) correct digits, from 0 to 9
y = zeros(10,60000); %Correct outputs vector
for i = 1:60000
    y(labels(i)+1,i) = 1;% if the correct digit is "i" then the correct 
                         % out layer is e_{i+1}
end

images = data(:,2:785); % the 28*28=784 pixel image is stored by row
images = images/255; % gray values from 0 to 255 are scaled

images = images'; %Input vectors

hn1 = 128; %Number of neurons in the first hidden layers
hn2=hn1; hn3=hn1; hn4=hn1;

%Initializing weights and biases
w12 = randn(hn1,784)*sqrt(2/784);
w23 = randn(hn2,hn1)*sqrt(2/hn1);
w34 = randn(hn3,hn2)*sqrt(2/hn2);
w45 = randn(hn4,hn3)*sqrt(2/hn3);
w56 = randn(10,hn4)*sqrt(2/hn4);

b12 = randn(hn1,1);
b23 = randn(hn2,1);
b34 = randn(hn3,1);
b45 = randn(hn4,1);
b56 = randn(10,1);

%learning rate
eta = 0.0058;

%Initializing errors and gradients
error6 = zeros(10,1);
error5 = zeros(hn4,1);
error4 = zeros(hn3,1);
error3 = zeros(hn2,1);
error2 = zeros(hn1,1);

errortot6 = zeros(10,1);
errortot5 = zeros(hn4,1);
errortot4 = zeros(hn3,1);
errortot3 = zeros(hn2,1);
errortot2 = zeros(hn1,1);

grad6 = zeros(10,1);
grad5 = zeros(hn4,1);
grad4 = zeros(hn3,1);
grad3 = zeros(hn2,1);
grad2 = zeros(hn1,1);

epochs = 50;

m = 10; %Minibatch size

% figure
% hold on
outerror = [];

for k = 1:epochs %Outer epoch loop
    
    batches = 1;
    
    for j = 1:60000/m
        error6 = zeros(10,1);
        error5 = zeros(hn4,1);
        error4 = zeros(hn3,1);
        error3 = zeros(hn2,1);
        error2 = zeros(hn1,1);
        
        errortot6 = zeros(10,1);
        errortot5 = zeros(hn4,1);
        errortot4 = zeros(hn3,1);
        errortot3 = zeros(hn2,1);
        errortot2 = zeros(hn1,1);
        
        grad6 = zeros(10,1);
        grad5 = zeros(hn4,1);
        grad4 = zeros(hn3,1);
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
    a4 = actfun(z4,af);
    z5 = w45*a4 + b45;
    a5 = actfun(z5,af);
    z6 = w56*a5 + b56;
%     a6 = actfun(z6,af);%Output vector
    a6 = actfun(z6,'Linear');%Output vector
    
    %backpropagation
%     error6 = (a6-y(:,i)).*actfunprime(z6,af);
    error6 = (a6-y(:,i)).*actfunprime(z6,'Linear');
    error5 = (w56'*error6).*actfunprime(z5,af);
    error4 = (w45'*error5).*actfunprime(z4,af);
    error3 = (w34'*error4).*actfunprime(z3,af);
    error2 = (w23'*error3).*actfunprime(z2,af);
    
    errortot6 = errortot6 + error6;
    errortot5 = errortot5 + error5;
    errortot4 = errortot4 + error4;
    errortot3 = errortot3 + error3;
    errortot2 = errortot2 + error2;
    
    grad6 = grad6 + error6*a5';
    grad5 = grad5 + error5*a4';
    grad4 = grad4 + error4*a3';
    grad3 = grad3 + error3*a2';
    grad2 = grad2 + error2*a1';

    end
    
    %Gradient descent
    
    w56 = w56 - eta/m*grad6;
    w45 = w45 - eta/m*grad5;
    w34 = w34 - eta/m*grad4;
    w23 = w23 - eta/m*grad3;
    w12 = w12 - eta/m*grad2;
    
    b56 = b56 - eta/m*errortot6;
    b45 = b45 - eta/m*errortot5;
    b34 = b34 - eta/m*errortot4;
    b23 = b23 - eta/m*errortot3;
    b12 = b12 - eta/m*errortot2;
    
    batches = batches + m;
    
    end
    
%     outerror = [outerror; norm(errortot2, inf)];
    fprintf('Epochs:');
    disp(k) %Track number of epochs
%     plot(outerror);
    [images,y] = shuffle(images,y); %Shuffles order of the images for next epoch
end

disp('Training done!')
%Saves the parameters
save(strcat(af,'_wsix_4_',num2str(hn1),'.mat'),'w56');
save(strcat(af,'_wfive_4_',num2str(hn1),'.mat'),'w45');
save(strcat(af,'_wfour_4_',num2str(hn1),'.mat'),'w34');
save(strcat(af,'_wthree_4_',num2str(hn1),'.mat'),'w23');
save(strcat(af,'_wtwo_4_',num2str(hn1),'.mat'),'w12');

save(strcat(af,'_bsix_4_',num2str(hn1),'.mat'),'b56');
save(strcat(af,'_bfive_4_',num2str(hn1),'.mat'),'b45');
save(strcat(af,'_bfour_4_',num2str(hn1),'.mat'),'b34');
save(strcat(af,'_bthree_4_',num2str(hn1),'.mat'),'b23');
save(strcat(af,'_btwo_4_',num2str(hn1),'.mat'),'b12');