clear all
clc

f_max = 40;
theta_0 = 90;
A = 1;
K = 8;
B = (f_max - A)/exp(K);
a = 6.34;
b = 10;
c = 0.5;
beta = 0.8;
lambda_0 = 10^(-2);
tau = 10;
% 
% A = 2.53;
% K = 8.08;
% B = 34.8/exp(K);

theta = linspace(-180,179,360).*2*pi/360;
figure(1)
plot(theta,tuning_curve(A,B,K,theta_0,theta))

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
    w2 = ifft(fft_w);
    w = [w(i + 180 +1:360),w(1:180 + i)];
    %noise = random('normal',0,0.1*mean(abs(w)),size(w));
    %w = w + noise;
    w_all = [w_all;w];
end
figure(2)
rng(2,'multFibonacci')
f_ini = abs(rand(360,1)); 
plot(theta,f_ini)
% x = -60:1:0;
% p = normpdf(x,-30,5);
% f_ini(60:120) = 3.*p;
% f_ini(90:150) = 2.5.*p;
u_ini = inv_sigmoid(a,b,c,beta,f_ini);
t_span = linspace(0,1000,1001);
[t,u] = ode45(@(t,u) diff_eqn(t,u,a,b,c,beta,w_all,tau) , t_span, u_ini);
% 
f = sigmoid(a,b,c,beta,u);

f1 = f(1001,:);
plot(theta,f1)