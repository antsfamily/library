close all;
clear all;

addpath ./../../../src
addpath ./../../../utils

filename = 'expt1';
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

%%% algorithm parameters
inneriters = 1;
outeriters = 250;
tol = 1e-4;

mu1 = 1;
mu2 = mu1;

%%%%%%%%%%% estimate noise using Median Absolute Deviation %%%%%%%%%%%
[ca, ch1, cv1, cd1] = dwt2(y,'db4');
sigma_est = sqrt( median(abs(cd1(:))) );

epsilon = sqrt(numel(y)+sqrt(8*numel(y)) )*sigma_est;

H2 = abs(H_FFT).^2;
tau = mu1/mu2;
filter_FFT = H2./(H2 + tau);
invLS = @(x, mu1) (1/mu1)*( x - WT( real(ifft2(filter_FFT.*fft2( W(x) ))) ) );

LS = @(x,mu) callcounter(invLS,x,mu);

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

calls_csalsa = calls;
calls = 0;

%%%%%%%%%%%% SPGl1
[M1, N1] = size(WTx);
[M2, N2] = size(Ax);

R = @(x, mode) A_wrapper(A, AT, x, M1, N1, M2, N2, mode);

options = spgSetParms('verbosity', 0, ...
                            'decTol',tol, ...
                            'iterations', 400);
fprintf('Running SPGl1...\n')

[x_spgl1,r,g,info] = spgl1_v0(R, reshape(y,M2*N2,1), 0, epsilon, [], options);
x_spgl1 = reshape(x_spgl1, [M1, N1]);
Wx_spgl1 = W(x_spgl1);
mse_spgl1 = norm(x-Wx_spgl1,'fro')^2 /(M*N);
ISNR_spgl1 = 10*log10( norm(y-x,'fro')^2 / (mse_spgl1*M*N) );

calls_spgl1 = calls;
calls = 0;

%%%%%%%%% NESTA
C = @(z) reshape(  A(reshape(z,[M1,N1])), [M2*N2,1] );
Ct = @(z) reshape( AT( reshape(z,[M2,N2]) ), [M1*N1,1]);

n = N;
counterA = 0;
counterAt = 0;

stop_th = tol;

U = @(z) z;
Ut = @(z) z;
mu = 1e-8;
opts = [];
opts.maxintiter = 5;
opts.TOlVar = stop_th;
opts.verbose = 0;
opts.maxiter = 2000;
opts.U = U;
opts.Ut = Ut;
opts.stoptest = 1;  
opts.typemin = 'L1';
opts.xplug = zeros( size(WTx(:)) );
opts.outFcn = @(z) [norm( W( reshape(z,[M1, N1] ) )-x, 'fro')^2/numel(x), ...
                    sum( abs( z(:) ) )];
delta = epsilon;
%counter();
% Ac = @(z) callcounter(C,z);
% Atc = @(z) callcounter(Ct,z);

fprintf('Running NESTA...\n')
tic;
[x_nesta,niter,resid, outData, times_nesta] = NESTA(C,Ct,y(:),mu,delta,opts);
t.NESTA = toc;
times_nesta = monotonize(times_nesta);
%NA_nesta = counter();
Xnesta = W(reshape(x_nesta,[M1,N1]));

mse_nesta = norm(Xnesta-x,'fro')^2/numel(x);
ISNR_nesta = 10*log10( norm(y-x,'fro')^2 / (mse_nesta*M*N) );

calls_nesta = calls;
calls = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('C-SALSA\nNumber of calls to A and AT: %d\n', calls_csalsa)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', length(times), times(end), mse, ISNR_final )

fprintf('SPGl1\nNumber of calls to A and AT: %d\n', calls_spgl1)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', length(info.times), info.times(end), mse_spgl1, ISNR_spgl1 )

fprintf('NESTA\nNumber of calls to A and AT: %d\n', calls_nesta )
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', niter, t.NESTA, mse_nesta, ISNR_nesta)


figure, imagesc(y), colormap gray, axis equal, axis off
figure, imagesc(Wz), colormap gray, axis equal, axis off, title('Estimate using C-SALSA');
figure, imagesc(Xnesta), colormap gray, axis equal, axis off, title('Estimate using NESTA');
figure, imagesc(Wx_spgl1), colormap gray, axis equal, axis off, title('Estimate using SPGL1');

teps = linspace(0, max(times_nesta(end), info.times(end)), 10);
epsbar = epsilon*ones(size(teps));

figure, semilogy(teps, epsbar, 'ks', ...
    times, criterion, 'b', info.times-info.times(1), info.criterion, 'c-.',...
    times_nesta, resid(:,1),'r:',...
    'LineWidth',3.5), 
legend('\epsilon', 'C-SALSA', 'SPGl1', 'NESTA'),
title('||A x^{k} - y||', 'FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');
ylabel('||A x^{k} - y||', 'FontName','Times','FontSize',14);


figure, semilogy(times, objective, info.times, info.obj, 'c-.',...
    times_nesta, outData(:,2), 'r:', 'LineWidth',3.5), 
legend('C-SALSA', 'SPGl1', 'NESTA', 'Location', 'SouthEast'),
title('Objective ||x^{k}||_{1}', 'FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');
ylabel('Objective ||x||_1', 'FontName','Times','FontSize',14);

