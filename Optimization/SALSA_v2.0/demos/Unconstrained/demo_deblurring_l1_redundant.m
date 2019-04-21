% Demo of image deblurring, with wavelets. 
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

B = @(x) real(ifft2(H_FFT.*fft2(x)));
BT = @(x) real(ifft2(HC_FFT.*fft2(x)));

%%%% wavelet representation
wav = daubcqf(2); % Haar wavelet
levels = 4;

W = @(x) mirdwt_TI2D(x, wav, levels); % inverse transform
WT = @(x) mrdwt_TI2D(x, wav, levels); % forward transform

%%%% true value (in wavelet representation)
WTx = WT(x);

%%%% function handles
A = @(x) B(W(x));
AT = @(x) WT(BT(x));

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);

%%%%% observation
Bx = B(x);
Psig  = norm(Bx,'fro')^2/(M*N);
BSNRdb = 40;
sigma = norm(Bx-mean(mean(Bx)),'fro')/sqrt(N*M*10^(BSNRdb/10));

y = Bx + sigma*randn(M,N);

%%%% algorithm parameters
%mu = 1e-3;
lambda = 0.0075; % reg parameter
mu = lambda;
muinv = 1/mu;
inneriters = 1;
outeriters = 150;
tol = 1e-3;

filter_FFT = HC_FFT./(abs(H_FFT).^2 + mu).*H_FFT;
invLS = @(x) muinv*( x - WT( real(ifft2(filter_FFT.*fft2( W(x) ))) ) );

invLS = @(x) callcounter(invLS,x);

fprintf('Running SALSA...\n')
[x_salsa, numA, numAt, objective, distance,  times, mses] = ...
         SALSA_v2(y, A, lambda,...
         'MU', mu, ...
         'AT', AT, ...
         'True_x', WTx, ...
         'ToleranceA', tol,...
         'MAXITERA', outeriters, ...
         'LS', invLS, ...
         'VERBOSE', 0);
     
Wx = W(x_salsa);
mse = norm(x-Wx,'fro')^2 /(M*N);
ISNR = 10*log10( norm(y-x,'fro')^2 / (mse*M*N) );
cpu_time = times(end);

%%%% display results and plots
fprintf('SALSA\nCalls = %d, iters = %d, CPU time = %3.3g seconds, \tFinal objective = %g, MSE = %3.3g, ISNR = %3.3g dB\n', ...
    calls, length(objective), cpu_time,  objective(end), mse, ISNR)

figure, imagesc(x), title('original'), colormap gray, axis equal, axis off
title('original')

figure, imagesc(y), colormap gray, axis equal, axis off
title('Blurred and noisy')

figure, imagesc(Wx), colormap gray, axis equal, axis off
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
