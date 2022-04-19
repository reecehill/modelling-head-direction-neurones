% function dfdt = diff_eqn(t,f,a,b,c,beta,w,tau)
%     
%     dfdt = ((1/b * log(exp((f/a).^(1/beta))-1) - c) + w*f) .* (a*b*beta*log(1+exp(b*((1/b * log(exp((f/a).^(1/beta))-1) - c)+c))).^(beta-1)) ./ (tau * (1+exp(-b*((1/b * log(exp((f/a).^(1/beta))-1) - c)+c))));
%     
% end
% 
% 



function dudt = diff_eqn(t,u,a,b,c,beta,w,tau)

    dudt = 1/tau * (-u + w*( a*(log(1 + exp(b*(u+c))).^beta) ));

end
