% Parameters
S0 = 100;
mu = 0.15;
sigma = 0.25;
N = 1000; % Large N for a fine partition
T = 1;
dt = T/N;
t = linspace(0, T, N+1);

% Preallocation
St_exact = zeros(1, N+1);
St_euler = zeros(1, N+1);
St_milstein = zeros(1, N+1);
St_exact(1) = S0;
St_euler(1) = S0;
St_milstein(1) = S0;

% Brownian increments
dW = sqrt(dt)*randn(1, N);

% Exact simulation
for i = 2:N+1
    St_exact(i) = S0*exp((mu - 0.5*sigma^2)*t(i) + sigma*sqrt(t(i))*randn);
end

% Euler-Maruyama and Milstein simulations
for i = 1:N
    St_euler(i+1) = St_euler(i) + mu*St_euler(i)*dt + sigma*St_euler(i)*dW(i);
    St_milstein(i+1) = St_milstein(i) + mu*St_milstein(i)*dt ...
                       + sigma*St_milstein(i)*dW(i) ...
                       + 0.5*sigma^2*St_milstein(i)*(dW(i)^2 - dt);
end

% Plotting
figure;
plot(t, St_exact, 'r', t, St_euler, 'b--', t, St_milstein, 'g:');
legend('Exact', 'Euler-Maruyama', 'Milstein');
title('Geometric Brownian Motion Simulation');
xlabel('Time');
ylabel('S_t');


% Parameters
num_sims = 10000; % Number of simulations for Monte Carlo
N_values = round(logspace(2, 4, 10)); % More values for a detailed plot

% Initialize arrays to store max errors
max_error_exp_euler = zeros(size(N_values));
max_error_exp_milstein = zeros(size(N_values));
max_error_path_euler = zeros(size(N_values));
max_error_path_milstein = zeros(size(N_values));

% Perform simulations
for j = 1:length(N_values)
    N = N_values(j);
    dt = T/N;
    t = linspace(0, T, N+1);

    % Simulation arrays
    S_exact = zeros(num_sims, N+1);
    S_euler = zeros(num_sims, N+1);
    S_milstein = zeros(num_sims, N+1);

    % Monte Carlo simulation
    for i = 1:num_sims
        dW = sqrt(dt)*randn(1, N); % Brownian increments
        W = [0, cumsum(dW)]; % Brownian path

        % Exact solution
        S_exact(i,:) = S0*exp((mu - 0.5*sigma^2)*t + sigma*W);

        % Euler-Maruyama
        S_euler(i,1) = S0;
        for k = 1:N
            S_euler(i,k+1) = S_euler(i,k) * (1 + mu*dt + sigma*dW(k));
        end

        % Milstein
        S_milstein(i,1) = S0;
        for k = 1:N
            S_milstein(i,k+1) = S_milstein(i,k) * (1 + mu*dt + sigma*dW(k) + 0.5*sigma^2*(dW(k)^2 - dt));
        end
    end

    % Calculate errors
    max_error_exp_euler(j) = max(abs(mean(S_euler) - mean(S_exact)));
    max_error_exp_milstein(j) = max(abs(mean(S_milstein) - mean(S_exact)));
    max_error_path_euler(j) = max(mean(abs(S_euler - S_exact)));
    max_error_path_milstein(j) = max(mean(abs(S_milstein - S_exact)));
end

% Generate log-log plots
figure;
loglog(N_values*dt, max_error_exp_euler, 'b-*', N_values*dt, max_error_exp_milstein, 'r-o');
hold on;
loglog(N_values*dt, max_error_path_euler, 'b--', N_values*dt, max_error_path_milstein, 'r-.');
legend('Euler Exp Error', 'Milstein Exp Error', 'Euler Path Error', 'Milstein Path Error');
xlabel('Delta t');
ylabel('Max Error');
title('Error Comparison in Log-Log Scale');
grid on;

% Parameters

num_sims = 10000; % Number of simulations for Monte Carlo
N_values = round(logspace(2, 4, 10)); % Different N values for different Delta t

% Initialize arrays to store expected maximum errors
error_M1_euler = zeros(length(N_values), 1);
error_M1_milstein = zeros(length(N_values), 1);

% Perform simulations
for j = 1:length(N_values)
    N = N_values(j);
    dt = T/N;
    M1_exact = zeros(num_sims, 1);
    M1_euler = zeros(num_sims, 1);
    M1_milstein = zeros(num_sims, 1);
    
    % Monte Carlo simulation
    for i = 1:num_sims
        dW = sqrt(dt)*randn(1, N); % Brownian increments
        W = [0, cumsum(dW)]; % Brownian path
        S = S0*exp((mu - 0.5*sigma^2)*(0:dt:T) + sigma*W);
        M1_exact(i) = max(S);
        
        S_euler = S0;
        S_milstein = S0;
        for k = 1:N
            S_euler = S_euler + mu * S_euler * dt + sigma * S_euler * dW(k);
            S_milstein = S_milstein + mu * S_milstein * dt ...
                        + sigma * S_milstein * dW(k) ...
                        + 0.5 * sigma^2 * (dW(k)^2 - dt);
            M1_euler(i) = max(M1_euler(i), S_euler);
            M1_milstein(i) = max(M1_milstein(i), S_milstein);
        end
    end
    
    % Calculate expected maximum error for both methods
    error_M1_euler(j) = abs(mean(M1_euler) - mean(M1_exact));
    error_M1_milstein(j) = abs(mean(M1_milstein) - mean(M1_exact));
end

% Generate log-log plots
figure;
loglog(N_values*dt, error_M1_euler, 'b-*', N_values*dt, error_M1_milstein, 'r-o');
xlabel('Delta t');
ylabel('Error in expected maximum');
title('Error in Running Maximum at Time 1');
legend('Euler-Maruyama', 'Milstein');
grid on;


% Initialize default count
nDefaults = 0;

% Monte Carlo simulation
for i = 1:N
    S = S0;
    for j = 1:nSteps
        dW = sqrt(dt)*randn;  % Brownian increment
        S = S + mu*S*dt + sigma*S*dW;  % GBM formula
        if S <= b
            nDefaults = nDefaults + 1;  % Count if default occurs
            break;  % Stop simulation for this path since default has occurred
        end
    end
end

% Calculate the default probability
P_default = nDefaults / N;

% Display the estimated default probability
fprintf('The estimated default probability is: %.4f\n', P_default);

   



