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


theta = linspace(-180,179,360).*2*pi/360;

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

    w_all = [w_all;w];

end
%% 

epsilon = 0.06;
stdev = mean(abs(w));
r = normrnd(0,stdev,360,360);
w_all_noisy = w_all + epsilon .* r;
final_states = [];
initial_states = [];
stable_states = [];
%% 

rng(2,'multFibonacci')
f_ini = abs(rand(360,1)); 
x = -60:1:-1;
p = 300.*normpdf(x,-30,5);

for i = 1:360
    if i > 301
        f_ini(i:360) = p(1:360-i+1);
        f_ini(1:60 - (360-i)-1) = p(360-i+2:60);
    
    else
        f_ini(i:i+59) = p;
    end
    initial_states = [initial_states ; f_ini];
    u_ini = inv_sigmoid(a,b,c,beta,f_ini);
    t_span = linspace(0,1000,1001);
    [t,u] = ode45(@(t,u) diff_eqn(t,u,a,b,c,beta,w_all,tau) , t_span, u_ini);
    f = sigmoid(a,b,c,beta,u);
    f1 = f(1001,:);
    stable_states = [stable_states ; f1];
%     hold on
%     plot(theta,f1)
    u_ini = inv_sigmoid(a,b,c,beta,f1);
    t_span = linspace(0,1000000,10001);
    [t,u] = ode45(@(t,u) diff_eqn(t,u,a,b,c,beta,w_all_noisy,tau) , t_span, u_ini);
    f = sigmoid(a,b,c,beta,u);
    f1 = f(10001,:);
    final_states = [final_states ; f1];
    hold on
    plot(theta,f1)
end
%% 

% u_ini = inv_sigmoid(a,b,c,beta,f1);
% t_span = linspace(0,1000000,1000001);
% [t,u] = ode45(@(t,u) diff_eqn(t,u,a,b,c,beta,w_all_noisy,tau) , t_span, u_ini);
% f = sigmoid(a,b,c,beta,u);
% f1 = f(1000001,:);
% plot(theta,f1)