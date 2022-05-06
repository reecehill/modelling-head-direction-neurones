clear all
clc

f_max = 40;
theta_0 = 0;
A = 1;
K = 8;
B = (f_max - A)/exp(K);
a = 6.34;
b = 10;
c = 0.5;
beta = 0.8;
lambda_0 = 10^(-3);
tau = 10;
% 
% A = 2.53;
% K = 8.08;
% B = 34.8/exp(K);

theta = linspace(-180,179,360).*2*pi/360;
b

w_all = [];

for i = -180:179
    
    theta_0 = i*2*pi/360;
    f_vals = tuning_curve(A,B,K,theta_0,theta);
    u_vals = inv_sigmoid(a,b,c,beta,f_vals);    
    fft_f = fft(f_vals);
    fft_u = fft(u_vals);
    fft_f_squared = abs(fft_f).^2;
    lambda = lambda_0 * max(fft_f_squared);
    
    fft_w = (fft_u .* fft_f)./(lambda + fft_f_squared);
    w = ifft(fft_w);
    w = [w(i + 180 +1:360),w(1:180 + i)];
    w = w;
    
    w_all = [w_all;w];
end

rng(2,'multFibonacci')
f_ini = abs(rand(360,1)); 
% plot(theta,f_ini)
x = -60:1:59;
p = normpdf(x,0,30).';
p_max = max(p);
scaling = 20/p_max;
f_ini(120:239) = scaling.*p;
% f_ini(181:300) = scaling.*p;
u_ini = inv_sigmoid(a,b,c,beta,f_ini);
t_span = linspace(0,1000,1001);
[t,u] = ode45(@(t,u) diff_eqn(t,u,a,b,c,beta,w_all,tau) , t_span, u_ini);
% 
f = sigmoid(a,b,c,beta,u);
f1 = f(1001,:);
f_vals = tuning_curve(A,B,K,0,theta);
hold on
xlabel("Theta (in degrees)")
xticks([-180,-90,0,90,180])
ylabel("Firing Rate (Hz)")
% zlabel("Firing Rate (in Hz)")
title("\lambda_{0} = 10^{-4}")
plot(theta.*(180/pi),f_vals)
plot(theta.*(180/pi),f1)
legend('Desired','Actual')

% hold on 
% grid on
% xlabel("Theta (in degrees)")
% ylabel("Time (in ms")
% zlabel("Firing Rate (in Hz)")
% title("Time Evoloution of Firing Rate")
% xticks([-180,-90,0,90,180])
% yticks(0:100:1000)
% zticks(0:5:40)

% for i = 1:11
%     f1 = f((i-1)*100 + 1,:);
%     plot3(theta.*(180/pi),(i-1)*100.*ones(length(theta),1),f1)
%     view([45 38])
%     
% end
