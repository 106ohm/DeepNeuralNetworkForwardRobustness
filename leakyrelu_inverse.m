function fr = leakyrelu_inverse(x)
    alpha = 0.01;
    f = zeros(length(x),1);
    for i = 1:length(x)
    if x(i)>=0
        f(i) = x(i);
    else
        f(i) = x(i)/alpha;
    end
    end
    fr = f;
end