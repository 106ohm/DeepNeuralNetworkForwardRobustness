function fr = leakyrelu(x)
    alpha = 0.01;
    f = zeros(length(x),1);
    for i = 1:length(x)
    if x(i)>=0
        f(i) = x(i);
    else
        f(i) = alpha*x(i);
    end
    end
    fr = f;
end