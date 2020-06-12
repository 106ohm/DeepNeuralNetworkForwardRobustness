function fr = actfunprime(x,af)
    switch af
        case 'elu'
            fr = elup(x);
        case 'ReLU'
            fr = relup(x);
        case 'LeakyReLU'
            fr = leakyrelup(x);
        otherwise
            error("activation function not implemented yet");
    end
end