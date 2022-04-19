function f = sigmoid(a,b,c,beta,x)

    f = a*(log(1+exp(b*(x+c))).^beta);
    
end