% Demo of image deblurring, with total variation. 
% 9*9 uniform blur, and Gaussian noise (SNR = 40 dB).

close all;
clear;
warning off all

addpath ./../../src
addpath ./../../utils

filename = 'expt1';

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
wav = daubcqf(4);
levels = 4;

W = @(x) mirdwt_TI2D(x, wav, levels); % inverse transform
WT = @(x) mrdwt_TI2D(x, wav, levels); % forward transform
% W= @(x) midwt(x, wav, levels); % inverse transform
% WT = @(x) mdwt(x, wav, levels); % forward transform
WTx = WT(x);

wav = daubcqf(2); % Haar wavelet
levels = 4;
P = @(x) mirdwt_TI2D(x, wav, levels); % inverse transform
PT = @(x) mrdwt_TI2D(x, wav, levels); % forward transform
% P = @(x) midwt(x, wav, levels); % inverse transform
% PT = @(x) mdwt(x, wav, levels); % forward transform


%%%% function handles
A = @(x) B(W(x));
AT = @(x) WT(BT(x));

%%%%% observation
Bx = B(x);
Psig  = norm(Bx,'fro')^2/(M*N);
BSNRdb = 40;
sigma = norm(Bx-mean(mean(Bx)),'fro')/sqrt(N*M*10^(BSNRdb/10));

y = Bx + sigma*randn(M,N);

ATy = AT(y);


%%%% phi = ||x||_1
tau1 =  0.008;
mu1 = tau1/5;

Phi = @(x) sum(abs(x(:)));
Psi = @(x, lambda) soft(x, lambda);

%%%% algorithm parameters
outeriters = 200;
tol = 1e-4;

filter_FFT1 = HC_FFT./(abs(H_FFT).^2 + mu1).*H_FFT;
invLS1 = @(x) ( x - WT( real(ifft2(filter_FFT1.*fft2( W(x) ))) ) )/mu1;

%%%% SALSA for synthesis prior only
fprintf('Running SALSA for synthesis prior only...\n')
[x_salsa1, numA, numAt, objective1, distance1, times1, mses1] = ...
         SALSA_v2(y, A, tau1,...
         'MU', mu1, ...
         'AT', AT, ...
         'StopCriterion', 1, ...
         'True_x', WTx, ...
         'ToleranceA', 1e-3,...
         'MAXITERA', 150, ...
         'LS', invLS1, ...
         'VERBOSE', 0);
Wx_salsa1 = W(x_salsa1);
mse1 = norm(x-Wx_salsa1,'fro')^2 /(M*N);
ISNR1 = 10*log10( norm(y-x,'fro')^2 / (mse1*M*N) );
cpu_time1 = times1(end);
fprintf('SALSA CPU time = %3.3g seconds, iters = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', cpu_time1, length(objective1)-1, mse1, ISNR1)
figure, imagesc(Wx_salsa1), colormap gray;
figure, semilogy(times1, objective1);
figure, semilogy(times1, mses1);


tau2 = 0.007;
mu2 = tau2*5;

filter_FFT2 = 1./(abs(H_FFT).^2 + mu2);
invLS2 = @(x) real(ifft2(filter_FFT2.*fft2( x )));

fprintf('Running SALSA for analysis prior only...\n')
[x_salsa2, numA, numAt, objective2, distance2, times2, mses2] = ...
         SALSA_v2(y, B, tau2,...
         'MU', mu2, ...
         'AT', BT, ...
         'StopCriterion', 1, ...
         'True_x', x, ...
         'ToleranceA', 1e-6,...
         'MAXITERA', 500, ...
         'P', P, ...
         'PT', PT, ...
         'LS', invLS2, ...
         'VERBOSE', 0);
mse2 = norm(x-x_salsa2,'fro')^2 /(M*N);
ISNR2 = 10*log10( norm(y-x,'fro')^2 / (mse2*M*N) );
cpu_time2 = times2(end);
fprintf('SALSA CPU time = %3.3g seconds, iters = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', cpu_time2, length(objective2)-1, mse2, ISNR2)
figure, imagesc(x_salsa2), colormap gray;
figure, semilogy(times2, objective2);
figure, semilogy(times2, mses2);


%%%%%%%%% compound regularization
tau1 = 0.0025;
mu1 = tau1/10;

tau2 = 0.007;
mu2 = tau2/10;

P2 = @(x) WT(P(x));
P2T = @(x) PT(W(x));

alpha = mu1 + mu2;
mu1inv = 1/mu1;
alphainv = 1/alpha;
filter_FFT = (abs(H_FFT).^2 + mu2)./(abs(H_FFT).^2 + alpha);
invLS = @(r) mu1inv*( r - WT( real(ifft2(filter_FFT.*fft2( W(r) ))) ) );

[x_cr, numA, numAt, objective, distance, times, mses_cr] = CoRAL_v2(y,A,tau1,tau2,...
    'AT', AT, ...
    'MAXITERA', 200, ...
    'MU1', mu1, ...
    'MU2', mu2, ...
    'P2', P2, ...
    'P2T', P2T, ...
    'STOPCRITERION', 1, ...
    'TOLERANCEA', tol, ...
    'LS', invLS,...
    'TRUE_X', WTx, ...
    'VERBOSE', 0 );
Wx_cr = W(x_cr);
mse_cr = norm(x-Wx_cr,'fro')^2 /(M*N);
ISNR_cr = 10*log10( norm(y-x,'fro')^2 / (mse_cr*M*N) );
cpu_time_cr = times(end);
fprintf('CoRAL\nCPU time = %3.3g seconds, iters = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', cpu_time_cr, length(objective)-1, mse_cr, ISNR_cr)
figure, imagesc(Wx_cr), colormap gray;
figure, semilogy(times, objective);
figure, semilogy(times, mses_cr);
