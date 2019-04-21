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

A = @(x) real(ifft2(H_FFT.*fft2(x)));
AT = @(x) real(ifft2(HC_FFT.*fft2(x)));

% denoising function;
chambolleit = 4;
Psi = @(x,th)  projk(x,th,chambolleit);
Phi = @(x) TVnorm(x);


%%%%% observation
BSNRdb = 40;
Ax = A(x);
sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(numel(Ax)*10^(BSNRdb/10));
y = Ax + sigma*randn(size(Ax));


global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);


%%%%%%%%%%% constrained SALSA

mu1 = 1;
mu2 = mu1;
epsilon = sqrt(N^2+8*sqrt(N^2))*sigma;
iters = 1000;
Pnum = norm(x(:)-y(:),2)^2;
tol = 1e-6;

H2 = abs(H_FFT).^2;
tau = mu1/mu2;
filter_FFT = H2./(H2 + tau);
invLS = @(x, mu1) (1/mu1)*( x - real( ifft2( filter_FFT.*fft2( x ) ) ) );

LS = @(x,mu) callcounter(invLS,x,mu);

fprintf('Running C-SALSA...\n')
[z, numA, numAt, objective, distance1, distance2, criterion, times, mses] = ...
         csalsa_v2(y, A, mu1, mu2, sigma,...
         'AT', AT, ...
         'PHI', Phi, ...
         'PSI', Psi, ...
         'StopCriterion', 1, ...
         'True_x', x, ...
         'ToleranceA', tol,...
         'MAXITERA', iters, ...
         'LS', LS, ...
         'VERBOSE', 0, ...
         'EPSILON', epsilon, ...
         'INITIALIZATION', 2, ...
         'TVINITIALIZATION', 1, ...
         'CONTINUATIONFACTOR', 1.01);
mse = norm(x-z,'fro')^2 /(M*N);
ISNR_final = 10*log10( norm(y-x,'fro')^2 / (mse*M*N) );
cpu_time = times(end);

calls_csalsa = calls;
calls = 0;

%%%%%%%%% NESTA
C = @(z) reshape(  A(reshape(z,[M,N])), [M*N,1] );
Ct = @(z) reshape( AT( reshape(z,[M,N]) ), [M*N,1]);

n = N;
stop_th = tol;
U = @(z) z;
Ut = @(z) z;
mu = 1e-6; %--- can be chosen to be small
opts = [];
opts.maxintiter = 5;
opts.TOlVar = stop_th;
opts.verbose = 0;
opts.maxiter = 1000;
opts.U = U;
opts.Ut = Ut;
opts.stoptest = 1;  
opts.typemin = 'tv';
opts.outFcn = @(z) [norm(z(:)-x(:),2)^2/numel(x), ...
                    TVnorm( reshape(z,[M,N]) )];
delta = epsilon;
% Ac = @(z) counter(C,z);
% Atc = @(z) counter(Ct,z);

fprintf('Running NESTA...\n')
tic;
[x_nesta,niter,resid,outData, times_nesta] = NESTA(C,Ct,y(:),mu,delta,opts);
t.NESTA = toc;
times_nesta = monotonize(times_nesta);
Xnesta = reshape(x_nesta,[M,N]);

mse_nesta = norm(Xnesta-x,'fro')^2/numel(x);
ISNR_nesta = 10*log10( norm(y-x,'fro')^2 / (mse_nesta*M*N) );

calls_nesta = calls;
calls = 0;

fprintf('C-SALSA\nNumber of calls to A and AT: %d\n', calls_csalsa)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', length(times), times(end), mses(end), ISNR_final )

fprintf('NESTA\nNumber of calls to A and AT: %d\n', calls_nesta)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', niter, t.NESTA, mse_nesta, ISNR_nesta)

figure, imagesc(y), colormap gray, axis equal, axis off
title('Blurred and noisy'),

figure, imagesc(z), colormap gray, axis equal, axis off
title('Constrained SALSA');

teps = linspace(0, times_nesta(end), 10);
epsbar = epsilon*ones(size(teps));

figure, semilogy(teps, epsbar, 'ks', ...
    times, criterion, 'b', ...
    times_nesta, resid(:,1),'r:',...
    'LineWidth',3.5), 
legend('\epsilon', 'C-SALSA', 'NESTA'),
title('||A x^{k} - y||', 'FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

figure, plot(times, objective, ...
    times_nesta, outData(:,2), 'r-.', 'LineWidth',3.5), 
legend('C-SALSA', 'NESTA', 'Location', 'SouthEast'),
title('Objective', 'FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

