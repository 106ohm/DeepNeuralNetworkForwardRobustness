test = load('mnist_test.csv');
labels = test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

af = 'LeakyReLU';

hn1=128;

images = test(:,2:785);
images = images/255;

images = images';

we56 = matfile(strcat(af,'_wsix_4_',num2str(hn1),'.mat'));
w6 = we56.w56;
we45 = matfile(strcat(af,'_wfive_4_',num2str(hn1),'.mat'));
w5 = we45.w45;
we34 = matfile(strcat(af,'_wfour_4_',num2str(hn1),'.mat'));
w4 = we34.w34;
we23 = matfile(strcat(af,'_wthree_4_',num2str(hn1),'.mat'));
w3 = we23.w23;
we12 = matfile(strcat(af,'_wtwo_4_',num2str(hn1),'.mat'));
w2 = we12.w12;

bi56 = matfile(strcat(af,'_bsix_4_',num2str(hn1),'.mat'));
b6 = bi56.b56;
bi45 = matfile(strcat(af,'_bfive_4_',num2str(hn1),'.mat'));
b5 = bi45.b45;
bi34 = matfile(strcat(af,'_bfour_4_',num2str(hn1),'.mat'));
b4 = bi34.b34;
bi23 = matfile(strcat(af,'_bthree_4_',num2str(hn1),'.mat'));
b3 = bi23.b23;
bi12 = matfile(strcat(af,'_btwo_4_',num2str(hn1),'.mat'));
b2 = bi12.b12;
success = 0;
n = 10000;

for i = 1:n
out2 = actfun(w2*images(:,i)+b2,af);
out3 = actfun(w3*out2+b3,af);
out4 = actfun(w4*out3+b4,af);
out5 = actfun(w5*out4+b5,af);
out = w6*out5+b6;
% out = actfun(w6*out5+b6,af);
if i == 37
    keyboard
end
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