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

calls_salsa = calls;
calls = 0;

fprintf('Running TwIST...\n')
[x_twist,dummy,obj_twist,time_twist,dummy,mse_twist] = TwIST(y, A, lambda, ...
          'AT', AT, ...
          'StopCriterion',3, ...
          'True_x', x, ...
          'ToleranceA',objective(end), ...
          'PSI', Psi_TV, ...
            'PHI', Phi_TV, ...
          'VERBOSE', 0);
twist_mse = norm(x-x_twist,'fro')^2 /(M*N);
ISNR_twist = 10*log10(  norm(y-x,'fro')^2 / (twist_mse*M*N) );
twist_time = time_twist(end);

calls_twist = calls;
calls = 0;

%%%% with SpaRSA
fprintf('Running SpaRSA...\n')
[x_sparsa,x_debias,objective_sparsa,times_sparsa,debias_start,mses_sparsa,taus]= ...
        SpaRSA(y,A,lambda,...
        'AT', AT, ...
          'StopCriterion',4, ...
          'True_x',x, ...
          'ToleranceA',objective(end), ...
          'Continuation', 0, ...
          'PSI', Psi_TV, ...
      'PHI', Phi_TV, ...
          'VERBOSE', 0);
sparsa_mse = norm(x-x_sparsa,'fro')^2 /(M*N);
ISNR_sparsa = 10*log10(  norm(y-x,'fro')^2 / (sparsa_mse*M*N) );
sparsa_time = times_sparsa(end);

calls_sparsa = calls;
calls = 0;

fprintf('Running FISTA...\n')
[X_out,objective_FISTA, times_FISTA, mses_FISTA] = my_deblur_fista(y,h,lambda, Phi_TV, Psi_TV, 3, objective(end), outeriters, x, 0);
mse_FISTA = norm(x-X_out,'fro')^2 /(M*N);
ISNR_FISTA = 10*log10( norm(y-x,'fro')^2 / (mse_FISTA*M*N) );
cpu_time_FISTA = times_FISTA(end);

calls_fista = calls;
calls = 0;

%%%% display results and plots
fprintf('TwIST CPU time = %3.3g seconds, iters = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', twist_time, length(time_twist), twist_mse, ISNR_twist)
fprintf('SpaRSA CPU time = %3.3g seconds, iters = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', sparsa_time, length(times_sparsa), sparsa_mse, ISNR_sparsa)
fprintf('FISTA CPU time = %3.3g seconds, iters = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', cpu_time_FISTA, length(objective_FISTA), mse_FISTA, ISNR_FISTA)
fprintf('SALSA\n CPU time = %3.3g seconds, iters = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', cpu_time, length(objective), mse, ISNR)

figure, imagesc(x), title('original'), colormap gray, axis equal, axis off
title('original')

figure, imagesc(y), colormap gray, axis equal, axis off
title('Blurred and noisy')

figure, imagesc(x_salsa), title('Estimated'), colormap gray, axis equal, axis off;
title('Estimated using SALSA')

figure, semilogy(time_twist, obj_twist, 'b', 'LineWidth',1.8), hold on, 
semilogy(times_FISTA, objective_FISTA,'g:', 'LineWidth',1.8),
semilogx(times_sparsa,objective_sparsa,'k-.', 'LineWidth',1.8),
semilogy(times, objective,'r--', 'LineWidth',1.8),
title('Objective function 0.5||y-Ax||_{2}^{2}+\lambda \Phi_{TV}(x)','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds'), 
legend('TwIST', 'FISTA', 'SpaRSA', 'SALSA');

