%% parameter setting
af = 'LeakyReLU';
alpha = 0.01;

hn1 = 80; %Number of neurons in the first hidden layer
hn2 = 60; %Number of neurons in the second hidden layer

delta = 0.01;
L = 5;

y = 7; % select digit (from 0 to 9)

precision = 3;

%% Start code

if y>=0 && y<=9
    y = y+1;
else
    error("y is %d, but it should be between 0 and 9", y);
end

%% load weights and biases
we34 = matfile(strcat(af,'_wfour.mat'));
w4 = we34.w34;
we23 = matfile(strcat(af,'_wthree.mat'));
w3 = we23.w23;
we12 = matfile(strcat(af,'_wtwo.mat'));
w2 = we12.w12;
bi34 = matfile(strcat(af,'_bfour.mat'));
b4 = bi34.b34;
bi23 = matfile(strcat(af,'_bthree.mat'));
b3 = bi23.b23;
bi12 = matfile(strcat(af,'_btwo.mat'));
b2 = bi12.b12;

out = zeros(10,1);
out(y) = 1;

% % this is the result for imag = 37 in digit_test_one_image.m
% out = [-0.0137
%    -0.0104
%    -0.0075
%    -0.0027
%    -0.0088
%    -0.0147
%    -0.0118
%     0.9942
%    -0.0081
%    -0.0048];

tic;
%% second layer
fprintf("Start second layer\n");
out2=pinv(w4)*(out-b4);
for i = 1 : hn2
    fprintf("Layer %d, %d-th neuron\n", 2, i);
    if out2(i)<0% definitely deactivated (case I)
        out2(i) = out2(i)/alpha;
        fprintf("case I\n");
        continue;
    elseif out2(i)>=0 % definitely activated (case II)
        % do nothing
        fprintf("case II\n");
        continue;
    else 
        error("no case III");
    end
end

%% first layer
fprintf("Start first layer\n");
out1=pinv(w3)*(out2-b3);
for i = 1 : hn1
    fprintf("Layer %d, %d-th neuron\n", 1, i);
    if out1(i)<0% definitely deactivated (case I)
        out1(i) = out1(i)/alpha;
        fprintf("case I\n");
        continue;
    elseif out1(i)>=0 % definitely activated (case II)
        % do nothing
        fprintf("case II\n");
        continue;
    else 
        error("no case III");
    end
end

%% 0-th (input) layer
fprintf("start 0-th layer\n");
in=pinv(w2)*(out1-b2);

toc;

%% save data
csvwrite('in_LeakyReLU.csv',in);
