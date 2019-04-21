% Demo of image inpainting (with 40% of the pixels missing), with total variation. 

close all;
clear;

addpath ./../../src
addpath ./../../utils

x = double(imread('cameraman.tif'));
[M, N] = size(x);

% random observations
O = rand(N)> 0.4; 
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

epsilon = sqrt(0.2)*sqrt(N^2+sqrt(8*N^2))*sigma;

tau = mu1/mu2;
LS = @(x, mu1) (1/mu1)*( x - (1/(1+tau))*A(x) );

fprintf('Running C-SALSA...\n')
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
         'TVINITIALIZATION', 1, ...
         'CONTINUATIONFACTOR', 1.18);
mse = norm(x-z,'fro')^2 /(M*N);
ISNR_final = 10*log10( norm(y-x,'fro')^2 / (mse*M*N) );
cpu_time = times(end);

%%%% display results
fprintf('C-SALSA\nNumber of calls to A and AT: %d\n', calls)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g, ISNR = %g dB\n', length(times), times(end), mses(end), ISNR_final )

figure, imagesc(y), colormap gray, axis equal, axis off
title('Blurred and noisy');

figure, imagesc(z), colormap gray, axis equal, axis off
title('Constrained SALSA');

epsbar = epsilon*ones(size(times));
figure, semilogy(times, epsbar,'r:', times, criterion, 'b', 'Linewidth', 2.4), 
title('||A x^{k}-y||','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('\epsilon', 'C-SALSA'),
xlabel('seconds');

figure, semilogy(times, mses, 'Linewidth', 1.8), 
title('MSE','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

figure, semilogy(times, objective, 'Linewidth', 1.8), 
title('Objective','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');
