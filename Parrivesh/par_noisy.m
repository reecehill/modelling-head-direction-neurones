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

% epsilon = 0.06;
% stdev = mean(abs(w));
% r = normrnd(0,stdev,360,360);
% w_all_noisy = w_all + epsilon .* r;
% final_states = zeros(360,360);
% initial_states = zeros(360,360);
% stable_states = zeros(360,360);
%% 

for ep = 6:8
    epsilon = 0.05 * ep;
    stdev = mean(abs(w));
    r = normrnd(0,stdev,360,360);
    w_all_noisy = w_all + epsilon .* r;
    final_states = zeros(360,360);
    initial_states = zeros(360,360);
    stable_states = zeros(360,360);
    rng(2,'multFibonacci') 
    x = -60:1:-1;
    p = 300.*normpdf(x,-30,5);
    
    parfor i = 1:360
        f_ini = abs(rand(360,1));
        if i > 301
            f_ini(i:360) = p(1:360-i+1);
            f_ini(1:60 - (360-i)-1) = p(360-i+2:60);
        
        else
            f_ini(i:i+59) = p;
        end
        initial_states(i,:) = f_ini;
        u_ini = inv_sigmoid(a,b,c,beta,f_ini);
        t_span = linspace(0,1000,1001);
        [t,u] = ode45(@(t,u) diff_eqn(t,u,a,b,c,beta,w_all,tau) , t_span, u_ini);
        f = sigmoid(a,b,c,beta,u);
        f1 = f(1001,:);
        stable_states(i,:) = f1;
    
        u_ini = inv_sigmoid(a,b,c,beta,f1);
        t_span = linspace(0,1000000,10001);
        [t,u] = ode45(@(t,u) diff_eqn(t,u,a,b,c,beta,w_all_noisy,tau) , t_span, u_ini);
        f = sigmoid(a,b,c,beta,u);
        f1 = f(10001,:);
        final_states(i,:) = f1;
        hold on
        plot(theta,f1)
    end
    save("Variables for Epsilon " + int2str(ep))

end
%% 
% hold on 
% for i = 1:360
%     plot(theta,final_states(i,:))
% end
% 
% c_theta = cos(theta);
% s_theta = sin(theta);
% 
% M_sum = sum(u_mat,2);
% xi_bar = u_mat*c_theta./M_sum;
% zeta_bar = u_mat*s_theta./M_sum;
% theta_max= unwrap(atan2(-zeta_bar,-xi_bar)+pi);

