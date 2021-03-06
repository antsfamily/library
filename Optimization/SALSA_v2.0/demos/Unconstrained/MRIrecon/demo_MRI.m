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
Psi_TV = @(x,th) projk(x,th,chambolleit); 
Phi_TV = @(x) TVnorm(real(x));

%%%% algorithm parameters
lambda = 9e-5;
mu = lambda*100;
inneriters = 1;
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
         'TVINITIALIZATION', 1, ...
         'StopCriterion', 1,...
       	 'ToleranceA', tol, ...
         'MAXITERA', outeriters, ...
         'LS', invLS, ...
         'Verbose', 0);
     
mse_salsa = norm(x- x_salsa,'fro')^2/numel(x);
time_salsa = times(end);

fprintf('SALSA\n Calls = %d, iters = %d, CPU time = %3.3g seconds, \tFinal objective = %g, MSE = %3.3g\n', ...
    calls, length(objective), time_salsa,  objective(end), mse_salsa)

figure, imagesc(mask), colormap gray, axis equal, axis off,
title('Sampling Mask (22 beams)');

figure, imagesc(real(x_salsa)), colormap gray, axis equal, axis off,
title('Estimated using SALSA');

figure, semilogy(times, objective, 'LineWidth',1.8),
title('Objective function 0.5||y-Ax||_{2}^{2}+\lambda \Phi_{TV}(x)','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

figure, semilogy(times, mses, 'LineWidth',1.8),
title('MSE','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');
