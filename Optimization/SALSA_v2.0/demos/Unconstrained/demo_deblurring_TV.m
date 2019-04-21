% Demo of image deblurring, with total variation. 
% 9*9 uniform blur, and Gaussian noise (SNR = 40 dB).

close all;
clear;

addpath ./../../src
addpath ./../../utils

%%%% original image
x = double( imread('cameraman.tif') );
[M, N] = size(x);

%%%% function handle for uniform blur operator (acts on the image
%%%% coefficients)
h = [1 1 1 1 1 1 1 1 1];
lh = length(h);
h = h/sum(h);
h = [h zeros(1,length(x)-length(h))];
h = cshift(h,-(lh-1)/2);
h = h'*h;

H_FFT = fft2(h);
HC_FFT = conj(H_FFT);

A = @(x) real(ifft2(H_FFT.*fft2(x)));
AT = @(x) real(ifft2(HC_FFT.*fft2(x)));

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);

%%%%% observation
Ax = A(x);
Psig  = norm(Ax,'fro')^2/(M*N);
BSNRdb = 40;
sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(N*M*10^(BSNRdb/10));

y = Ax + sigma*randn(M,N);

%%%% algorithm parameters
lambda = 2.5e-2; % reg parameter
mu = lambda/10;
outeriters = 500;
tol = 1e-5;

%%%% TV regularization
Phi_TV = @(x) TVnorm(x);
chambolleit = 5;
Psi_TV = @(x,th)  mex_vartotale(x,th,'itmax',chambolleit); 

filter_FFT = 1./(abs(H_FFT).^2 + mu);
invLS = @(x) real(ifft2(filter_FFT.*fft2( x )));

invLS = @(x) callcounter(invLS,x);

fprintf('Running SALSA...\n')
[x_salsa, numA, numAt, objective, distance, times, mses] = ...
         SALSA_v2(y, A, lambda,...
         'MU', mu, ...
         'AT', AT, ...
         'StopCriterion', 1, ...
         'True_x', x, ...
         'ToleranceA', tol,...
         'MAXITERA', outeriters, ...
         'Psi', Psi_TV, ...
         'Phi', Phi_TV, ...
         'TVINITIALIZATION', 1, ...
         'TViters', 10, ...
         'LS', invLS, ...
         'VERBOSE', 0);

mse = norm(x-x_salsa,'fro')^2 /(M*N);
ISNR = 10*log10( norm(y-x,'fro')^2 / (mse*M*N) );
cpu_time = times(end);

%%%% display results and plots

fprintf('SALSA\n Calls = %d, iters = %d, CPU time = %3.3g seconds, \tFinal objective = %g, MSE = %3.3g, ISNR = %3.3g dB\n', ...
    calls, length(objective), cpu_time,  objective(end), mse, ISNR)

figure, imagesc(x), title('original'), colormap gray, axis equal, axis off
title('original')

figure, imagesc(y), colormap gray, axis equal, axis off
title('Blurred and noisy')

figure, imagesc(x_salsa), title('Estimated'), colormap gray, axis equal, axis off;
title('Estimated')

figure, semilogy(times, objective, 'LineWidth',1.8),
title('Objective function 0.5||y-Ax||_{2}^{2}+\lambda||x||_{1}','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

figure, semilogy(times, mses, 'LineWidth',1.8),
title('MSE','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');
