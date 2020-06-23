
%% parameter setting
af = 'ReLU';
hn1 = 128; %Number of neurons in each hidden layer
hn2=hn1; hn3=hn1; hn4=hn1;

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
%% load weights and biases
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
        out1(i) = 0;
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
%         [~, low] = fmincon(fun_min,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
        if up<0
            up = -up;
        end
        count = count + 1;
        s.u(count) = up;
%         s.l(count) = low;
        s.l(count) = 0;
        out1(i) = sym(strcat('e',int2str(1),'_',int2str(i)));
        s.syms = [s.syms; out1(i)];
        assume([0 <= s.syms(count), s.syms(count) <= up]);
    end
end

csvwrite(strcat('out1_ReLU_4_',num2str(hn1),'.csv'),char(out1));

%% second layer
fprintf("Start second layer\n");
out2 = w3*out1+b3;
out2=vpa(out2,precision);
for i = 1 : hn2
    fprintf("Layer %d, %d-th neuron\n", 2, i);
    if isAlways(out2(i)<0)% definitely deactivated (case I)
        out2(i) = 0;
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
%         [~, low] = fmincon(fun_min,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
        if up<0
            up = -up;
        end
        count = count + 1;
        s.u(count) = up;
%         s.l(count) = low;
        s.l(count) = 0;
        out2(i) = sym(strcat('e',int2str(2),'_',int2str(i)));
        s.syms = [s.syms; out2(i)];
        assume([0 <= s.syms(count), s.syms(count) <= up]);
    end
end

csvwrite(strcat('out2_ReLU_4_',num2str(hn1),'.csv'),char(out2));

%% third layer
fprintf("Start third layer\n");
out3 = w4*out2+b4;
out3=vpa(out3,precision);
for i = 1 : hn3
    fprintf("Layer %d, %d-th neuron\n", 3, i);
    if isAlways(out3(i)<0)% definitely deactivated (case I)
        out3(i) = 0;
        continue;
    elseif isAlways(out3(i)>=0) % definitely activated (case II)
        % do nothing
        continue;
    else % (caseIII)
        v = symvar(out3(i));
        f = matlabFunction(out3(i),'vars', v);
        g = gradient(out3(i));
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        u = [];
        l = [];
        nonlcon = [];
        for k = 1 : length(v)
%             fprintf('Layer %d, %d-th neuron, k=%d\n', 3, i, k);
            indx = find(ismember(s.syms, v(k))==1);
            if isempty(indx) || length(indx)>1
                error("third layer: troubles with the index (needs to exists and be unique). k=%d, v(k)=%s",k,v(k));
            end
            u = [u, s.u(indx)];
            l = [l, s.l(indx)];
        end
        fun_max = @(x)apply_minus_fun(f,x,g);
        fun_min = @(x)apply_fun(f,x,g);
        [~, up] = fmincon(fun_max,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
%         [~, low] = fmincon(fun_min,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
        if up<0
            up = -up;
        end
        count = count + 1;
        s.u(count) = up;
%         s.l(count) = low;
        s.l(count) = 0;
        out3(i) = sym(strcat('e',int2str(3),'_',int2str(i)));
        s.syms = [s.syms; out3(i)];
        assume([0 <= s.syms(count), s.syms(count) <= up]);
    end
end

csvwrite(strcat('out3_ReLU_4_',num2str(hn1),'.csv'),char(out3));

%% forth layer
fprintf("Start forth layer\n");
out4 = w5*out3+b5;
out4=vpa(out4,precision);
for i = 1 : hn4
    fprintf("Layer %d, %d-th neuron\n", 4, i);
    if isAlways(out4(i)<0)% definitely deactivated (case I)
        out4(i) = 0;
        continue;
    elseif isAlways(out4(i)>=0) % definitely activated (case II)
        % do nothing
        continue;
    else % (caseIII)
        v = symvar(out4(i));
        f = matlabFunction(out4(i),'vars', v);
        g = gradient(out4(i));
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        u = [];
        l = [];
        nonlcon = [];
        for k = 1 : length(v)
%             fprintf('Layer %d, %d-th neuron, k=%d\n', 4, i, k);
            indx = find(ismember(s.syms, v(k))==1);
            if isempty(indx) || length(indx)>1
                error("forth layer: troubles with the index (needs to exists and be unique). k=%d, v(k)=%s",k,v(k));
            end
            u = [u, s.u(indx)];
            l = [l, s.l(indx)];
        end
        fun_max = @(x)apply_minus_fun(f,x,g);
        fun_min = @(x)apply_fun(f,x,g);
        [~, up] = fmincon(fun_max,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
%         [~, low] = fmincon(fun_min,zeros(length(v),1),A,b,Aeq,beq,l,u,nonlcon,options);
        if up<0
            up = -up;
        end
        count = count + 1;
        s.u(count) = up;
%         s.l(count) = low;
        s.l(count) = 0;
        out4(i) = sym(strcat('e',int2str(4),'_',int2str(i)));
        s.syms = [s.syms; out4(i)];
        assume([0 <= s.syms(count), s.syms(count) <= up]);
    end
end

csvwrite(strcat('out4_ReLU_4_',num2str(hn1),'.csv'),char(out4));

%% fifth (and last) layer
fprintf("start fifth layer\n");
out = w6*out4+b6;
out=vpa(out,precision);
final_upper = zeros(10,1);
final_lower = zeros(10,1);
for i = 1 : 10
    fprintf("Layer %d, %d-th neuron\n", 5, i);
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
%         fprintf('Layer %d, %d-th neuron, k=%d\n', 5, i, k);
        indx = find(ismember(s.syms, v(k))==1);
        if isempty(indx) || length(indx)>1
            error("fifth layer: troubles with the index (needs to exists and be unique)");
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
    final_upper(i) = up;
    final_lower(i) = low;
end

toc;

%% save data
csvwrite(strcat('final_bounds_ReLU_4_',num2str(hn1),'.csv'),[final_lower,final_upper]);

csvwrite(strcat('out_ReLU_4_',num2str(hn1),'.mat'),char(out));

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