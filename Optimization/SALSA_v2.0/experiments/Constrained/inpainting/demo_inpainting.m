
close all;
clear all;

%addpath ./Measurements
filename = 'demo_missing_pixels';
addpath ./../../../src
addpath ./../../../utils

global numA;
global numAt;

%x = double(imread('pirate.tif'));
x = double(imread('cameraman.tif'));
%x = double(imread('phantom.tif'));
[M, N] = size(x);

% random observations
O = rand(N)> 0.4;    % 50%

y = x.* O;

% set BSNR
BSNR = 40;
Py   = var(x(:));
sigma = sqrt((Py/10^(BSNR/10)));
% add noise
y = O.*(y + sigma*randn(N));

% handle functions for TwIST
%  convolution operators
A = @(x)  O.*x;
AT = @(x) O.*x;

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);

% denoising function;
Psi_TV = @(x,th)  projk(x,th,16);
% TV regularizer;
Phi_TV = @(x) TVnorm(x);

Pnum = norm(y(:)-x(:),2)^2/numel(x);

tolA = 1e-4;
outeriters = 80;

mu1 = 0.01;
mu2 = 1;

tau = mu1/mu2;
epsilon = sqrt(0.2)*sqrt(N^2+sqrt(8*N^2))*sigma;

tau = mu1/mu2;
LS = @(x, mu1) (1/mu1)*( x - (1/(1+tau))*A(x) );

fprintf('Running C-SALSA...\n')
tic
[z, numA, numAt, objective, distance1, distance2, criterion, times, mses] = ...
         csalsa_v2(y, A, mu1, mu2, sigma,...
         'AT', AT, ...
         'PHI', Phi_TV, ...
         'PSI', Psi_TV, ...
         'StopCriterion', 1, ...
         'True_x', x, ...
         'ToleranceA', tolA,...
         'MAXITERA', outeriters, ...
         'LS', LS, ...
         'VERBOSE', 0, ...
         'EPSILON', epsilon, ...
         'INITIALIZATION', 2, ...
         'CONTINUATIONFACTOR', 1.18);
toc
mse = norm(x-z,'fro')^2 /numel(x);
ISNR_final = 10*log10( norm(y-x,'fro')^2 / (mse*numel(x)) );
cpu_time = times(end);

fprintf('Number of calls to A and AT: %d\n', calls)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', length(times), times(end), mses(end), ISNR_final )

calls_csalsa = calls;
calls = 0;

%%%%%%%%%% NESTA
C = @(z) reshape(  A(reshape(z,[N,N])), [N*N,1] );
Ct = @(z) reshape( AT( reshape(z,[N,N]) ), [N*N,1]);

n = N;

% global counterA;
% global counterAt;
% 
% counterA = 0;
% counterAt = 0;

stop_th = tolA;

%fprintf('2) run NESTA\n\n');
U = @(z) z;
Ut = @(z) z;
mu = 1e-6; %--- can be chosen to be small
opts = [];
opts.maxintiter = 5;
opts.TOlVar = stop_th;
opts.verbose = 0;
opts.maxiter = 5000;
opts.U = U;
opts.Ut = Ut;
opts.stoptest = 1;  
opts.typemin = 'tv';
opts.outFcn = @(z) [norm(z(:)-x(:),2)^2/numel(x), ...
                    TVnorm( reshape(z,[N,N]) )];
delta = epsilon;
% Ac = @(z) counter(C,z);
% Atc = @(z) counter(Ct,z);

fprintf('Running NESTA...\n')
tic
[x_nesta,niter,resid,outData, times_nesta] = NESTA(C,Ct,y(:),mu,delta,opts);
t.NESTA = toc
times_nesta = monotonize(times_nesta);
Xnesta = reshape(x_nesta,[N,N]);

calls_nesta = calls;
calls = 0;

mse_nesta = norm(Xnesta-x,'fro')^2/numel(x);
ISNR_nesta = 10*log10( norm(y-x,'fro')^2 / (mse_nesta*N*N) );


fprintf('SALSA\nNumber of calls to A and AT: %d\n', calls_csalsa)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', length(times), times(end), mses(end), ISNR_final )

fprintf('NESTA\nNumber of calls to A and AT: %d\n', calls_nesta)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', niter, t.NESTA, mse_nesta, ISNR_nesta)

figure, imagesc(y), colormap gray, axis equal, axis off
title('Blurred and noisy');

figure, imagesc(z), colormap gray, axis equal, axis off
title('Estimate using Constrained SALSA');

figure, imagesc(Xnesta), colormap gray, axis equal, axis off,
title('Estimate using NESTA)');

teps = linspace(0, max(times_nesta(end), times(end)), 10);
epsbar = epsilon*ones(size(teps));
figure, plot(teps, epsbar,'ks', times, criterion, 'b', times_nesta, resid(:,1), 'r-.', 'Linewidth', 3.5), 
title('||A x^{k}-y||','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('\epsilon', 'C-SALSA', 'NESTA', 'Location', 'SouthEast'),
xlabel('seconds');

figure, semilogy(times, mses, times_nesta, outData(:,1), 'r-.', 'Linewidth', 3.5), 
title('MSE','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('C-SALSA', 'NESTA'),
xlabel('seconds');

figure, semilogy(times, objective, times_nesta, outData(:,2), 'r:', 'Linewidth', 3.5), 
title('Objective','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('C-SALSA', 'NESTA'),
xlabel('seconds');
