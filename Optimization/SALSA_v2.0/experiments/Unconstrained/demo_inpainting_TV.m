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
Psi_TV = @(x,th)  projk(real(x),th,chambolleit);
% TV regularizer;
Phi_TV = @(x) TVnorm(x);

%%%% parameters
lambda = 0.2*sigma^2;
mu = 5e-3;
tolA = 1e-4;
outeriters = 500;

%%%% SALSA

invATA = O*(1/(1+mu))+(1-O)*(1/mu);
invLS = @(x) invATA.*x;

invLS = @(x) callcounter(invLS,x);

%%%% SALSA
fprintf('Running SALSA...\n')
[x_salsa, numA, numAt, objective, distance,  times, mses]= ...
         SALSA_v2(y,A,lambda,...
         'MU', mu, ...
         'AT', AT, ...
         'True_x', x,...       
         'TVINITIALIZATION', 0, ...
         'Psi', Psi_TV, ...
         'Phi', Phi_TV, ...
         'StopCriterion', 1,...
       	 'ToleranceA', tolA,...
         'Initialization',y,...
         'MAXITERA', outeriters, ...
         'LS', invLS, ...
         'Verbose', 0);

mse_salsa = norm(x- x_salsa,'fro')^2/numel(x);
ISNR_salsa = 10*log10( sum((y(:)-x(:)).^2)./(mse_salsa*numel(x)) );
time_salsa = times(end);

calls_salsa = calls;
calls = 0;

%%%%% -- TwIST ---------------------------
% stop criterium:  the relative change in the objective function 
% falls below 'ToleranceA'
fprintf('Running TwIST...\n')
[x_twist,dummy,obj_twist,...
    times_twist,dummy,mses_twist]= ...
         TwIST(y,A,lambda,...
         'AT', AT, ...
         'lambda',1e-3, ...
         'True_x', x,...       
         'Psi', Psi_TV, ...
         'Phi',Phi_TV, ...
         'Initialization',y,...
         'StopCriterion', 3,...
       	 'ToleranceA', objective(end),...
         'MAXITERA', outeriters, ...
         'Verbose', 0);
mse_twist = norm(x- x_twist,'fro')^2/numel(x);
ISNR_twist = 10*log10( sum((y(:)-x(:)).^2)./(mse_twist*numel(x)) );
time_twist = times_twist(end);

calls_twist = calls;
calls = 0;

%%%% FISTA
L = 1;
fprintf('Running FISTA...\n')
[X_out,objective_FISTA, times_FISTA, mses_FISTA] = my_fista(y,A,AT,lambda,L, Phi_TV, Psi_TV, 3, objective(end), outeriters, x, 0);
mse_FISTA = norm(x-X_out,'fro')^2 /(N*N);
ISNR_FISTA = 10*log10( norm(y-x,'fro')^2 / (mse_FISTA*N*N) );
time_FISTA = times_FISTA(end);

calls_fista = calls;
calls = 0;

%%%% display results
fprintf('TwIST CPU time = %3.3g seconds, calls = %d \titers = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', time_twist, calls_twist, length(times_twist), mse_twist, ISNR_twist)
fprintf('FISTA CPU time = %3.3g seconds, calls = %d \titers = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', time_FISTA, calls_fista, length(objective_FISTA), mse_FISTA, ISNR_FISTA)
fprintf('SALSA\nCPU time = %3.3g seconds, calls = %d \titers = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', time_salsa, calls_salsa, length(objective), mse_salsa, ISNR_salsa)

figure, colormap(gray), imagesc(x), axis equal, axis off, title('Original');

figure, colormap gray, imagesc(y), axis equal, axis off, title('Missing Samples - 40%');
 
figure, imagesc(x_salsa), colormap gray, axis equal, axis off, title('Restored Image - SALSA');

figure, plot(times_twist, obj_twist, 'b', 'LineWidth',1.8), hold on, 
plot(times_FISTA, objective_FISTA,'g:', 'LineWidth',1.8),
plot(times, objective,'r--', 'LineWidth',1.8),
title('Objective function 0.5||y-Ax||_{2}^{2}+\lambda \Phi_{TV}(x)','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds'), 
legend('TwIST', 'FISTA', 'SALSA');

