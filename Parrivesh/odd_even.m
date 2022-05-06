% clear all
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
gamma = -2*pi/100;
alpha = 0.0037;
% 
% A = 2.53;
% K = 8.08;
% B = 34.8/exp(K);

theta = linspace(-180,179,360).*2*pi/360;
%plot(theta,tuning_curve(A,B,K,theta_0,theta))

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
    w = [w(i + 180 +1:360),w(1:180 + i)] + (alpha.*sin(theta-theta_0))*2*pi/360;

    w_all = [w_all;w];
end

rng(2,'multFibonacci')
% f_ini = abs(rand(360,1)); 
f_ini = tuning_curve(A,B,K,0,theta);

u_ini = inv_sigmoid(a,b,c,beta,f_ini);
% t_span = linspace(0,1000,1001);
% [t,u] = ode45(@(t,u) diff_eqn(t,u,a,b,c,beta,w_all,tau) , t_span, u_ini);
% % 
% f = sigmoid(a,b,c,beta,u);
% f1 = f(1001,:);
% f_ini = f1;
% u_ini = inv_sigmoid(a,b,c,beta,f_ini);

% t_span = linspace(0,1000,1001);
t_span = linspace(0,400,401);

[t,u] = ode45(@(t,u) diff_eqn(t,u,a,b,c,beta,w_all,tau) , t_span, u_ini);
% 
f = sigmoid(a,b,c,beta,u);
% f1 = f(1001,:);
% plot(theta,f1)
hold on 
grid on
xlabel("Theta (in degrees)")
ylabel("Time (in ms)")
zlabel("Firing Rate (in Hz)")
title("Time Evoloution of Firing Rate")
xticks([-180,-90,0,90,180])
yticks(0:50:400)
zticks(0:5:40)
view([45 38])
for i = 1:9
    f1 = f((i-1)*50 + 1,:);
    plot3(theta.*(180/pi),(i-1)*50.*ones(length(theta),1),f1)
end
%% 
% hold on
% ylim([-0.4,0.4])
% yticks(-0.4:0.2:0.4)
% xticks([-180,-90,0,90,180])
% % plot(theta.*180/pi,(alpha.*sin(theta-theta_0))*2*pi)
% plot(theta.*180/pi,w_all(181,:)*360)