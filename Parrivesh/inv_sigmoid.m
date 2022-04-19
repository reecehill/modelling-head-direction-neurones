function out = inv_sigmoid(a,b,c,beta,f)

    out = 1/b * log(exp((f/a).^(1/beta))-1) - c;

end

