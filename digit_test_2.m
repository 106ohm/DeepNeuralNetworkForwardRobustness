test = load('mnist_test.csv');
labels = test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

af = 'LeakyReLU';

images = test(:,2:785);
images = images/255;

images = images';

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
success = 0;
n = 10000;

for i = 1:n
out2 = actfun(w2*images(:,i)+b2,af);
out3 = actfun(w3*out2+b3,af);
out = actfun(w4*out3+b4,af);
big = 0;
num = 0;
for k = 1:10 % the entry of out with largest value is taken as the answer
    if out(k) > big
        num = k-1;
        big = out(k);
    end
end

if labels(i) == num
    success = success + 1;
end
    

end

fprintf('Accuracy: ');
fprintf('%f',success/n*100);
disp(' %');