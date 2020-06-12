function fr = actfun(x,af)
    switch af
        case 'elu'
            fr = elu(x);
        case 'ReLU'
            fr = relu(x);
        case 'LeakyReLU'
            fr = leakyrelu(x);
        case 'LeakyReLU_inverse'
            fr = leakyrelu_inverse(x);
        case 'Linear'
            fr = linear(x);
        otherwise
            error("activation function not implemented yet");
    end
end