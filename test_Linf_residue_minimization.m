A = rand(10,60);
b = rand(10,1);

options = optimset('MaxIter',1000);

x0 = zeros(60,1);

fun = @(x)apply_fun(x,A,b);

x = fminsearch(fun,x0);

function r = apply_fun(x,A,b)
    r = norm(A*x-b,inf);
end