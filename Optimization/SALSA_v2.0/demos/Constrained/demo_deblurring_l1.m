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

%%%% wavelet representation
wav = daubcqf(2); % Haar wavelet
levels = 4;

W = @(x) mirdwt_TI2D(x, wav, levels); % inverse transform
WT = @(x) mrdwt_TI2D(x, wav, levels); % forward transform

B = @(x) real(ifft2(H_FFT.*fft2(x)));
BT = @(x) real(ifft2(HC_FFT.*fft2(x)));

A = @(x) B(W(x));
AT = @(x) WT(BT(x));

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);


H2 = abs(H_FFT).^2;

%%%%% observation
BSNRdb = 40;
Ax = B(x);
sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(numel(Ax)*10^(BSNRdb/10));
y = Ax + sigma*randn(size(Ax));

Pnum = norm(x(:)-y(:),2)^2;

WTx = WT(x);

%%%%%%%%%% constrained SALSA

%%%% algorithm parameters
inneriters = 1;
outeriters = 250;
tol = 1e-4;
mu1 = 1;
mu1inv = 1/mu1;
mu2 = mu1;


H2 = abs(H_FFT).^2;
tau = mu1/mu2;
filter_FFT = H2./(H2 + tau);
invLS = @(x, mu1) (1/mu1)*( x - WT( real(ifft2(filter_FFT.*fft2( W(x) ))) ) );

LS = @(x,mu) callcounter(invLS,x,mu);

%%%%%%%%%%% estimate noise using Median Absolute Deviation %%%%%%%%%%%
[ca, ch1, cv1, cd1] = dwt2(y,'db4');
sigma_est = sqrt( median(abs(cd1(:))) );

epsilon = sqrt(N^2+8*sqrt(N^2))*sigma_est;

fprintf('Running C-SALSA...\n')
[z, numA, numAt, objective, distance1, distance2, criterion, times, mses] = ...
         csalsa_v2(y, A, mu1, mu2, sigma,...
         'AT', AT, ...
         'StopCriterion', 2, ...
         'True_x', WTx, ...
         'ToleranceA', tol,...
         'MAXITERA', outeriters, ...
         'LS', LS, ...
         'VERBOSE', 0, ...
         'EPSILON', epsilon, ...
         'CONTINUATIONFACTOR', 1.03);
Wz = W(z);
mse = norm(x-Wz,'fro')^2 /(M*N);
ISNR_final = 10*log10( norm(y-x,'fro')^2 / (mse*M*N) );
cpu_time = times(end);


%%%% display results and plots
fprintf('C-SALSA\nNumber of calls to A and AT: %d\n', calls)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', length(times), times(end), mse, ISNR_final )

figure, imagesc(y), colormap gray, axis equal, axis off
title('Blurred and noisy');

figure, imagesc(Wz), colormap gray, axis equal, axis off
title('Constrained SALSA');

epsbar = epsilon*ones(size(times));
figure, semilogy(times, epsbar, 'r:', times, criterion, 'b', 'LineWidth',1.8), 
legend('\epsilon', 'SALSA'),
title('||A x^{k} - y||', 'FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

figure, semilogy(times, objective, 'LineWidth',1.8), 
title('Objective ||x^{k}||_{1}', 'FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

