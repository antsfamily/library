% Demo of image inpainting (with 40% of the pixels missing), with total variation. 

close all;
clear;

addpath ./../../src
addpath ./../../utils

x = double(imread('cameraman.tif'));
N=length(x);

% random observations
O = rand(N)> 0.4;    % 40% missing
y= x.* O;

% set BSNR
BSNR = 40;
Py = var(x(:));
sigma= sqrt((Py/10^(BSNR/10)));
% add noise
y=y+ sigma*randn(N);
y = y.*O;

% handle functions for TwIST
%  convolution operators
A = @(x)  O.*x;
AT = @(x) O.*x;

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);

% denoising function;
chambolleit = 20;
Psi_TV = @(x,th)  mex_vartotale(x,th,'itmax',chambolleit); 
%Psi_TV = @(x,th,px,py)  chambolle_prox_TV_stop(real(x), 'lambda', th, 'maxiter', 5, 'dualvars',[px py]);
% TV regularizer;
Phi_TV = @(x) TVnorm(x);

%%%% parameters
lambda = 0.2*sigma^2;
mu = 5e-3;
tolA = 1e-4;
outeriters = 300;

invATA = O*(1/(1+mu))+(1-O)*(1/mu);
invLS = @(x) invATA.*x;

LS = @(x) callcounter(invLS,x);

%%%% SALSA
fprintf('Running SALSA...\n')
[x_salsa, numA, numAt, objective, distance,  times, mses]= ...
         SALSA_v2(y,A,lambda,...
         'MU', mu, ...
         'AT', AT, ...
         'True_x', x,...       
         'TVINITIALIZATION', 1, ...
         'Psi', Psi_TV, ...
         'Phi', Phi_TV, ...
         'StopCriterion', 1,...
       	 'ToleranceA', tolA,...
         'Initialization',y,...
         'MAXITERA', outeriters, ...
         'LS', LS, ...
         'Verbose', 0);

mse_salsa = norm(x- x_salsa,'fro')^2/numel(x);
ISNR_salsa = 10*log10( sum((y(:)-x(:)).^2)./(mse_salsa*numel(x)) );
time_salsa = times(end);

Pnum = sum((y(:)-x(:)).^2);
ISNR_seq = 10*log10(Pnum./(mses*numel(x)));

%%%% display results
fprintf('SALSA\n Calls = %d, iters = %d, CPU time = %3.3g seconds, \tFinal objective = %g, MSE = %3.3g, ISNR = %3.3g dB\n', ...
    calls, length(objective), time_salsa,  objective(end), mse_salsa, ISNR_salsa)

figure, colormap(gray), imagesc(x), axis equal, axis off, title('Original');

figure, colormap gray, imagesc(y), axis equal, axis off, title('Missing Samples - 40%');
 
figure, imagesc(x_salsa), colormap gray, axis equal, axis off, title('Restored Image - SALSA');

figure, semilogy(times, objective, 'LineWidth',1.8),
title('Objective function 0.5||y-Ax||_{2}^{2}+\lambda \Phi_{TV}(x)','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

figure, plot(times, ISNR_seq, 'LineWidth',1.8),
title('Evolution of the ISNR (dB)','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('time (seconds)');
