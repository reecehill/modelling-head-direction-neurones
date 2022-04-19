function f = tuning_curve(A,B,K,theta_0,theta)

    f = A + B*exp(K*cos(theta-theta_0));

end