function fr = relu(x)
    f = zeros(length(x),1);
    for i = 1:length(x)
    if x(i)>=0
        f(i) = x(i);
    else
        f(i) = 0;
    end
    end
    fr = f;
end