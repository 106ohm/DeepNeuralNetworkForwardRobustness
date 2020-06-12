%% parameter setting
af = 'LeakyReLU';
alpha = 0.01;

hn1 = 80; %Number of neurons in the first hidden layer
hn2 = 60; %Number of neurons in the second hidden layer

imag = 37;% index of test image

delta = 0.01;
L = 5;

precision = 3;

% optimization parameters
options = optimoptions('fmincon','SpecifyObjectiveGradient',true);

%% load test
test = load('mnist_test.csv');
% identify (supervised) correct answers
label = test(imag,1);
y = zeros(10,1);
y(label+1) = 1;

%% load image
image = test(imag,2:785);
image = image/255;
image = image';

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

%% define struct with symbols and bounds
% init the struct only with input layer related informations
count = 784;
s = struct();
s.syms = sym('e0_%d', [count, 1], 'real');
s.u = zeros(count,1);
s.l = zeros(count,1);
for i=1:784
    s.l(i) = image(i) - delta;
    s.u(i) = image(i) + delta;
    assume([image(i) - delta <= s.syms(i), s.syms(i) <= image(i) + delta]);
end

%% input layer
out0 = s.syms;

tic;
%% first layer
fprintf("Start first layer\n");
out1 = w2*out0+b2;
out1=vpa(out1,precision);
for i = 1 : hn1
    fprintf("Layer %d, %d-th neuron\n", 1, i);
    if isAlways(out1(i)<0)% definitely deactivated (case I)
        out1(i) = alpha*out1(i);
        continue;
    elseif isAlways(out1(i)>=0) % definitely activated (case II)
        % do nothing
        continue;
    else % (caseIII)
        v = symvar(out1(i));
        f = matlabFunction(out1(i),'vars', v);
        g = gradient(out1(i));
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        u = [];
        l = [];
        nonlcon = [];
        for k = 1 : length(v)
%             fprintf('Layer %d, %d-th neuron, k=%d\n', 2, i, k);
            indx = find(ismember(s.syms, v(k))==1);
            if isempty(indx) || length(indx)>1
                error("fisrt layer: troubles with the index (needs to exists and be unique)");
            end
            u = [u, s.u(indx)];
            l = [l, s.l(indx)];
        end
        fun_max = @(x)apply_minus_fun(f,x,g);
        fun_min = @(x)apply_fun(f,x,g);
        [~, up] = fmincon(fun_max,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
        [~, low] = fmincon(fun_min,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
        if up<0
            up = -up;
        end
	if low>0 && abs(low)>2*delta
	    error("low is %f, but should be less than zero", low);
	end
        count = count + 1;
        s.u(count) = up;
        s.l(count) = alpha*low;
        s.l(count) = 0;
        out1(i) = sym(strcat('e',int2str(1),'_',int2str(i)));
        s.syms = [s.syms; out1(i)];
        assume([alpha*low <= s.syms(count), s.syms(count) <= up]);
    end
end

csvwrite('out1_LeakyReLU',char(out1));

%% second layer
fprintf("Start second layer\n");
out2 = w3*out1+b3;
out2=vpa(out2,precision);
for i = 1 : hn2
    fprintf("Layer %d, %d-th neuron\n", 2, i);
    if isAlways(out2(i)<0)% definitely deactivated (case I)
        out2(i) = alpha*out2(i);
        continue;
    elseif isAlways(out2(i)>=0) % definitely activated (case II)
        % do nothing
        continue;
    else % (caseIII)
        v = symvar(out2(i));
        f = matlabFunction(out2(i),'vars', v);
        g = gradient(out2(i));
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        u = [];
        l = [];
        nonlcon = [];
        for k = 1 : length(v)
%             fprintf('Layer %d, %d-th neuron, k=%d\n', 2, i, k);
            indx = find(ismember(s.syms, v(k))==1);
            if isempty(indx) || length(indx)>1
                error("second layer: troubles with the index (needs to exists and be unique). k=%d, v(k)=%s",k,v(k));
            end
            u = [u, s.u(indx)];
            l = [l, s.l(indx)];
        end
        fun_max = @(x)apply_minus_fun(f,x,g);
        fun_min = @(x)apply_fun(f,x,g);
        [~, up] = fmincon(fun_max,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
        [~, low] = fmincon(fun_min,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
        if up<0
            up = -up;
        end
	if low>0 && abs(low)>2*delta
	    error("low is %f, but should be less than zero", low);
	end
        count = count + 1;
        s.u(count) = up;
        s.l(count) = alpha*low;
        s.l(count) = 0;
        out2(i) = sym(strcat('e',int2str(2),'_',int2str(i)));
        s.syms = [s.syms; out2(i)];
        assume([alpha*low <= s.syms(count), s.syms(count) <= up]);
    end
end

csvwrite('out2_LeakyReLU',char(out2));

%% third (and last) layer
fprintf("start third layer\n");
out = w4*out2+b4;
out=vpa(out,precision);
final_upper = zeros(10,1);
final_lower = zeros(10,1);
for i = 1 : 10
    fprintf("Layer %d, %d-th neuron\n", 3, i);
    v = symvar(out(i));
    f = matlabFunction(out(i),'vars', v);
    g = gradient(out(i));
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    u = [];
    l = [];
    nonlcon = [];
    
    for k = 1 : length(v)
%         fprintf('Layer %d, %d-th neuron, k=%d\n', 3, i, k);
        indx = find(ismember(s.syms, v(k))==1);
        if isempty(indx) || length(indx)>1
            error("third layer: troubles with the index (needs to exists and be unique)");
        end
        u = [u, s.u(indx)];
        l = [l, s.l(indx)];
    end
    fun_max = @(x)apply_minus_fun(f,x,g);
    fun_min = @(x)apply_fun(f,x,g);
    [~, up] = fmincon(fun_max,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
    [~, low] = fmincon(fun_min,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
    up = -up;
    final_upper(i) = up;
    final_lower(i) = low;
    if final_lower(i)<0
        final_lower(i) = alpha*final_lower(i);
    end
    if final_upper(i)<0
        final_upper(i) = alpha*final_upper(i);
    end
end

toc;

%% save data
csvwrite('final_bounds_LeakyReLU.csv',[final_lower,final_upper]);

csvwrite('out_LeakyReLU',char(out));

function [value, grad] = apply_fun (f, x, g)
    args = num2cell(x);
    value = f(args{:});
    
    if nargout > 1 % gradient required
        grad = double(g);
    end
end

function [value, grad] = apply_minus_fun (f, x, g)
    args = num2cell(x);
    value = -f(args{:});
    
    if nargout > 1 % gradient required
        grad = -double(g);
    end
end
