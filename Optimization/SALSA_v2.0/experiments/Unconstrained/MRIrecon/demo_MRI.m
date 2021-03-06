% Demo of MRI image reconstruction, with total variation. 
% 128*128 SHepp-Logan phantom, 22 radial lines.

close all;
clear;

addpath ./../../../src
addpath ./../../../utils

N = 128;

x = phantom(N);

angles = 22;

[mask_temp,Mh,mi,mhi] = LineMask(angles,N);
mask = fftshift(mask_temp);
A = @(x)  masked_FFT(x,mask);
AT = @(x) (masked_FFT_t(x,mask));
ATA = @(x) (ifft2c(mask.*fft2c(x))) ;

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);
ATA = @(x) callcounter(ATA,x);

sigma =  1e-3/sqrt(2);
y = A(x);
y = y + sigma*(randn(size(y)) + i*randn(size(y)));

%%%% TV regularization
chambolleit = 40;
Psi_TV = @(x,th)  projk(real(x),th,chambolleit);
Phi_TV = @(x) TVnorm(real(x));

%%%% algorithm parameters
lambda = 9e-5;
mu = lambda*100;
inneriters = 5;
outeriters = 1000;
tol = 5e-6;

invLS = @(x) (x - (1/(1+mu))*ATA(x) )/mu;

fprintf('Running SALSA...\n')
[x_salsa, numA, numAt, objective, distance,  times, mses]= ...
         SALSA_v2(y,A,lambda,...
         'AT', AT, ...
         'Mu', mu, ...
         'Psi', Psi_TV, ...
         'Phi', Phi_TV, ...
         'True_x', x,...       
         'TVINITIALIZATION', 0, ...
         'StopCriterion', 1,...
       	 'ToleranceA', tol, ...
         'MAXITERA', outeriters, ...
         'LS', invLS, ...
         'Verbose', 0);

mse_salsa = norm(x- x_salsa,'fro')^2/numel(x);
time_salsa = times(end);

calls_salsa = calls;
calls = 0;

fprintf('Running TwIST...\n')
[x_twist, dummy, objective_twist, times_twist, debias_start, mses_twist]= ...
         TwIST(y,A,lambda,...
         'AT', AT, ...
         'True_x', x,...       
         'Psi', Psi_TV, ...
         'Phi', Phi_TV, ...
         'StopCriterion', 3,...
       	 'ToleranceA', objective(end), ...
         'MaxiterA', outeriters, ...
         'Verbose', 0);
mse_twist = norm(x- x_twist,'fro')^2/numel(x);
time_twist = times_twist(end);
fprintf('TwIST:\nIters = %d, CPU time = %g seconds, MSE = %g\n', length(objective_twist), time_twist, mse_twist)

calls_twist = calls;
calls = 0;

fprintf('Running SpaRSA...\n')
[x_sparsa, dummy, objective_sparsa, times_sparsa, debias_start, mses_sparsa]= ...
         SpaRSA(y,A,lambda,...
         'AT', AT, ...
         'True_x', x,...       
         'Psi', Psi_TV, ...
         'Phi', Phi_TV, ...
         'StopCriterion', 4,...
       	 'ToleranceA', objective_twist(end), ...
         'MaxiterA',  outeriters, ...
         'Verbose', 0);
mse_sparsa = norm(x- x_sparsa,'fro')^2/numel(x);
time_sparsa = times_sparsa(end);
fprintf('SpaRSA:\nIters = %d, CPU time = %g seconds, MSE = %g\n', length(objective_sparsa), time_sparsa, mse_sparsa)

calls_sparsa = calls;
calls = 0;

fprintf('Running FISTA...\n')
[x_fista,objective_FISTA, times_FISTA, mses_FISTA] = my_fista(y,A, AT,lambda, 2.0506, Phi_TV, Psi_TV, 3, objective_twist(end), outeriters, x, 0);
mse_FISTA = norm(x-x_fista,'fro')^2 /numel(x);
time_FISTA = times_FISTA(end);
calls_fista = calls;
calls = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('TwIST:\nIters = %d, calls = %d, CPU time = %g seconds, MSE = %g\n', length(objective_twist), calls_twist, time_twist, mse_twist)
fprintf('SpaRSA:\nIters = %d, calls = %d, CPU time = %g seconds, MSE = %g\n', length(objective_sparsa), calls_sparsa, time_sparsa, mse_sparsa)
fprintf('FISTA:\nIters = %d, calls = %d, CPU time = %g seconds, MSE = %g\n', length(objective_FISTA), calls_fista, time_FISTA, mse_FISTA)
fprintf('SALSA:\nIters = %d, calls = %d, CPU time = %g seconds, MSE = %g\n', length(objective), calls_salsa, time_salsa, mse_salsa)

figure, imagesc(mask), colormap gray, axis equal, axis off,
title('Sampling Mask (22 beams)');

figure, imagesc(real(x_salsa)), colormap gray, axis equal, axis off,
title('Estimated using SALSA');


figure, semilogy(times_sparsa, objective_sparsa,'g-.', 'LineWidth',1.8), hold on, 
semilogy(times_twist, objective_twist, 'b', 'LineWidth',1.8),
semilogy(times_FISTA, objective_FISTA,'m:', 'LineWidth',1.8),
semilogy(times, objective,'r--', 'LineWidth',1.8),
title('Objective function 0.5||y-Ax||_{2}^{2}+\lambda \Phi_{TV}(x)','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds'), 
legend('SpaRSA', 'TwIST', 'FISTA', 'SALSA');

figure, semilogy(times_sparsa,mses_sparsa,'k-.', 'LineWidth',1.8), hold on, 
semilogy(times_twist, mses_twist, 'b', 'LineWidth',1.8),
semilogy(times_FISTA, mses_FISTA,'m:', 'LineWidth',1.8),
semilogy(times, mses,'r--', 'LineWidth',1.8),
title('MSE','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds'), 
legend('SpaRSA', 'TwIST', 'FISTA', 'SALSA');

