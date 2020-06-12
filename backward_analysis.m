%% parameter setting
af = 'LeakyReLU';
alpha = 0.01;

hn1 = 80; %Number of neurons in the first hidden layer
hn2 = 60; %Number of neurons in the second hidden layer

delta = 0.01;
L = 5;

y = 7; % select digit (from 0 to 9)

precision = 3;

% optimization parameters
options = optimoptions('fmincon','SpecifyObjectiveGradient',true);

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

%% define struct with symbols and bounds
% init the struct only with output layer related informations
count = 10;
s = struct();
s.syms = sym('e3_%d', [count, 1], 'real');
s.u = zeros(count,1);
s.l = zeros(count,1);
for i=1:10
    if i==y
        s.l(i) = 1 - delta;
        s.u(i) = 1 + delta;
        assume([1 - delta <= s.syms(i), s.syms(i) <= 1 + delta]);
    else
        s.l(i) = 0;
        s.u(i) = 0;
        assume(s.syms(i)==0);
    end
end

%% output layer
out = s.syms;

csvwrite('backward_out_LeakyReLU',char(out));

tic;
%% second layer
fprintf("Start second layer\n");
out2=pinv(w4)*(out-b4);
out2=vpa(out2,precision);
for i = 1 : hn2
    fprintf("Layer %d, %d-th neuron\n", 2, i);
    if isAlways(out2(i)<0)% definitely deactivated (case I)
        out2(i) = out2(i)/alpha;
        fprintf("case I\n");
        continue;
    elseif isAlways(out2(i)>=0) % definitely activated (case II)
        % do nothing
        fprintf("case II\n");
        continue;
    else % (caseIII)
        fprintf("case III\n");
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
                error("second layer: troubles with the index (needs to exists and be unique)");
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
        s.l(count) = low/alpha;
        s.l(count) = 0;
        out2(i) = sym(strcat('e',int2str(2),'_',int2str(i)));
        s.syms = [s.syms; out2(i)];
        assume([low/alpha <= s.syms(count), s.syms(count) <= up]);
    end
end

csvwrite('backward_out2_LeakyReLU',char(out2));

%% first layer
fprintf("Start first layer\n");
out1=pinv(w3)*(out2-b3);
out1=vpa(out1,precision);
for i = 1 : hn1
    fprintf("Layer %d, %d-th neuron\n", 1, i);
    if isAlways(out1(i)<0)% definitely deactivated (case I)
        out1(i) = out1(i)/alpha;
        fprintf("case I\n");
        continue;
    elseif isAlways(out1(i)>=0) % definitely activated (case II)
        % do nothing
        fprintf("case II\n");
        continue;
    else % (caseIII)
        fprintf("case III\n");
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
%             fprintf('Layer %d, %d-th neuron, k=%d\n', 1, i, k);
            indx = find(ismember(s.syms, v(k))==1);
            if isempty(indx) || length(indx)>1
                error("first layer: troubles with the index (needs to exists and be unique). k=%d, v(k)=%s",k,v(k));
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
        s.l(count) = low/alpha;
        s.l(count) = 0;
        out1(i) = sym(strcat('e',int2str(1),'_',int2str(i)));
        s.syms = [s.syms; out1(i)];
        assume([low/alpha <= s.syms(count), s.syms(count) <= up]);
    end
end

csvwrite('backward_out1_LeakyReLU',char(out1));

%% 0-th (input) layer
fprintf("start 0-th layer\n");
in=pinv(w2)*(out1-b2);
in=vpa(in,precision);
initial_upper = zeros(784,1);
initial_lower = zeros(784,1);
for i = 1 : 784
    fprintf("Layer %d, %d-th neuron\n", 0, i);
    v = symvar(in(i));
    f = matlabFunction(in(i),'vars', v);
    g = gradient(in(i));
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    u = [];
    l = [];
    nonlcon = [];
    
    for k = 1 : length(v)
%         fprintf('Layer %d, %d-th neuron, k=%d\n', 0, i, k);
        indx = find(ismember(s.syms, v(k))==1);
        if isempty(indx) || length(indx)>1
            error("0-th layer: troubles with the index (needs to exists and be unique)");
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
    initial_upper(i) = up;
    initial_lower(i) = low;
%     if initial_lower(i)<0
%         initial_lower(i) = initial_lower(i)/alpha;
%     end
%     if initial_upper(i)<0
%         initial_upper(i) = initial_upper(i)/alpha;
%     end
end

toc;

%% save data
csvwrite('initial_bounds_LeakyReLU.csv',[initial_lower,initial_upper]);

%% plot
mesh(reshape(initial_lower,[28,28]));

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
