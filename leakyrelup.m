function fr = leakyrelup(x)
    alpha = 0.01;
    f = zeros(length(x),1);
    for i = 1:length(x)
    if x(i)>=0
        f(i) = 1;
    else
        f(i) = alpha;
    end
    end
    fr = f;
end