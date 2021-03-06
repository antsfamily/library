close all;
clear all;

addpath ./../../../src
addpath ./../../../utils

filename = 'analysis_expt1';
verbose = 0;

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

WTx = WT(x);

B = @(x) real(ifft2(H_FFT.*fft2(x)));
BT = @(x) real(ifft2(HC_FFT.*fft2(x)));

A = @(x) B(x);
AT = @(x) BT(x);

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);

%%%%% observation
BSNRdb = 40;
Ax = B(x);
sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(numel(Ax)*10^(BSNRdb/10));
y = Ax + sigma*randn(size(Ax));

Pnum = norm(x(:)-y(:),2)^2;

%%%%%%%%%%% constrained SALSA
tol = 1e-4;
mu1 = 1.2;
mu2 = mu1;
tau = mu1/mu2;
epsilon = sqrt(N^2+8*sqrt(N^2))*sigma;
iters = 250;
delta = 1.03;

H2 = abs(H_FFT).^2;
tau = mu1/mu2;
filter_FFT = H2./(H2 + tau);
invLS = @(x, mu1) (1/mu1)*( x - real( ifft2( filter_FFT.*fft2( x ) ) ) );

LS = @(x,mu) callcounter(invLS,x,mu);

fprintf('Running C-SALSA...\n')
[z, numA, numAt, objective, distance1, distance2, criterion, times, mses] = ...
         csalsa_v2(y, A, mu1, mu2, sigma,...
         'AT', AT, ...
         'P', W, ...
         'PT', WT, ...
         'StopCriterion', 2, ...
         'True_x', x, ...
         'ToleranceA', tol,...
         'MAXITERA', iters, ...
         'LS', LS, ...
         'VERBOSE', 0, ...
         'EPSILON', epsilon, ...
         'INITIALIZATION', 2, ...
         'CONTINUATIONFACTOR', delta);
mse = norm(x-z,'fro')^2 /(M*N);
ISNR_final = 10*log10( norm(y-x,'fro')^2 / (mse*M*N) );
cpu_time = times(end);

calls_csalsa = calls;
calls = 0;

[M1, N1] = size(WTx);
[M2, N2] = size(Ax);


%%%%%%%%% NESTA
C = @(z) reshape(  B(reshape(z,[M,N])), [M2*N2,1] );
Ct = @(z) reshape( BT( reshape(z,[M2,N2]) ), [M*N,1]);


n = N;
% global counterA;
% global counterAt;
% 
% counterA = 0;
% counterAt = 0;

stop_th = tol;

U = @(z) reshape( WT( reshape(z,[M,N]) ), [M1*N1, 1]);
Ut = @(z) reshape( W( reshape(z,[M1,N1]) ), [M*N,1]);
mu = 1e-8; %--- can be chosen to be small
opts = [];
opts.maxintiter = 5;
opts.TOlVar = stop_th;
opts.verbose = 0;
opts.maxiter = 5000;
opts.U = U;
opts.Ut = Ut;
opts.stoptest = 1;  
opts.typemin = 'L1';
opts.outFcn = @(z) [norm(z-x(:),2)^2/numel(x),...
                    norm( reshape( WT( reshape(z,[M,N]) ), [M1*N1,1]) ,1)];
delta = epsilon;
% counter();
% Ac = @(z) wrapper_Acount(C,z);
% Atc = @(z) wrapper_Atcount(Ct,z);

fprintf('Running NESTA...\n')
tic;
[x_nesta,niter,resid, outData, times_nesta] = NESTA(C,Ct,y(:),mu,delta,opts);
t.NESTA = toc;
times_nesta= monotonize(times_nesta);
%NA_nesta = counter();
Xnesta = reshape(x_nesta,[M,N]);

calls_nesta = calls;
calls = 0;

mse_nesta = norm(Xnesta-x,'fro')^2/numel(x);
ISNR_nesta = 10*log10( norm(y-x,'fro')^2 / (mse_nesta*M*N) );


fprintf('C-SALSA\nNumber of calls to A and AT: %d\n', calls_csalsa)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', length(times), times(end), mse, ISNR_final )

fprintf('NESTA\nNumber of calls to A and AT: %d\n', calls_nesta)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', niter, t.NESTA, mse_nesta, ISNR_nesta)

figure, imagesc(y), colormap gray, axis equal, axis off
title('Blurred and noisy'),

figure, imagesc(z), colormap gray, axis equal, axis off
title('Estimate using C-SALSA');

teps = linspace(0, max(times_nesta(end), times(end)), 10);
epsbar = epsilon*ones(size(teps));

figure, semilogy(teps, epsbar, 'kx',...
    times, criterion, 'b', times_nesta, resid(:,1),'r:',...
    'LineWidth',1.8), 
legend('\epsilon', 'C-SALSA', 'NESTA'),
title('||A x^{k} - y||', 'FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

figure, semilogy(times, objective, times_nesta, outData(:,2), 'r:', 'Linewidth', 1.8), 
title('||W^T x^k||_{1}','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('C-SALSA', 'NESTA', 'Location', 'SouthEast'),
xlabel('seconds'), 
