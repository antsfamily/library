%  Based on DemoTV.m from the NESTA package by Jerome Bobin, Caltech
%
%  This is a short script that performs comparisons between NESTA and RecPF
%  in the scope of TV minimization recovery from partial Fourier measurements 
% 
%  min ||x||_TV s.t. ||b - A x||_2 <= epsilon
%
%  Here A*A is assumed to be an orthogonal projector
%
%  Parameters to set : 1) n : the size of the image to reconstruct (default: 128)
%                      2) Dyna : the dynamic range of the original image (in dB) (default : 40)
%
%  Created entities : I : original image
%                     Xnesta : image recovered with Nesta
%                     Xrpf_default : image recovered with RecPF using the default parameters
%                     Xrpf : image recovered with RecPF using the tuned parameters
%                     NA_X = number of calls of A or A* for the method (X = nesta, nesta_chg, rpf, rpf_default)
%

close all;
clear all;

addpath ./../../../src
addpath ./../../../utils
addpath ../MRIrecon


verbose = 0;

n = 128;   %--- The data are n*n images
Dyna = 40; %--- Dynamic range in dB

%%%% Setting up the experiment
N = n*n;
x = MakeRDSquares(n,7,Dyna);

L = floor(55*(n/256));
[M,Mh,mi,mhi] = LineMask(L,n);
OMEGA = mhi;
OMEGA = [OMEGA];

K = length(OMEGA);

B = @(z) B_fhp(z,OMEGA);
BT = @(z) Bt_fhp(z,OMEGA,n);

b0 = B(x);
sigma = 0.1;
noise = sigma*randn(size(b0));
y = b0+noise;

x_bp = BT(y);

%%%%% common parameters
epsilon = sqrt(numel(y)+8*sqrt(numel(y)))*sigma;
stop_th = 1e-5;
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% NESTA

A = @(z) A_fhp(z,OMEGA);
At = @(z) At_fhp(z,OMEGA,n);

global calls;
calls = 0;
A = @(x) callcounter(A,x);
At = @(x) callcounter(At,x);

b0 = A(reshape(x,n*n,1));
sigma = 0.1;
noise = sigma*randn(size(b0));
b = b0+noise;

fprintf('\n\nTotal Variation minimization experiment\n\nImage size = %g x %g / Dynamic range : %g / Noise level : %g\n\n',n,n,Dyna,sigma);

U = @(z) z;
Ut = @(z) z;
mu = 0.2; %--- can be chosen to be small
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
                    TVnorm( reshape(z,[n,n]) )];


fprintf('Running NESTA...\n')
tic;
[x_nesta,niter,resid,outData, times_nesta] = NESTA(A,At,y,mu,epsilon,opts);
t.NESTA = toc;
times_nesta = monotonize(times_nesta);
Xnesta = reshape(x_nesta,n,n);

mse_nesta = norm(Xnesta(:)-x(:),2)^2/numel(x);

calls_nesta = calls;
calls = 0;

%%%%%%%%% C-SALSA

%%%% TV regularization
chambolleit = 8;
Psi_TV = @(x,th)  projk(real(x),th,chambolleit);
Phi_TV = @(x) TVnorm(real(x));

fprintf('Running C-SALSA...\n')

mu1 = 0.15;
mu2 = mu1;
tau = mu1/mu2;

delta = 1.2;
iters = 500;

tau = mu1/mu2;
LS = @(x,mu1) (1/mu1)*(x - BT( (1/(tau+1))*B(x)));

calls = 0;
B = @(x) callcounter(B,x);
BT = @(x) callcounter(BT,x);

[x_salsa, numA, numAt, objective, distance1, distance2, criterion, times, mses] = ...
         csalsa_v2(y, B, mu1, mu2, sigma,...
         'AT', BT, ...
         'StopCriterion', 2, ...
         'ToleranceA', stop_th,...
         'MAXITERA', iters, ...
         'TRUE_X', x, ...
         'PHI', Phi_TV, ...
         'PSI', Psi_TV, ...
         'LS', LS, ...
         'VERBOSE', 0, ...
         'EPSILON', epsilon, ...
         'INITIALIZATION', 2, ...
         'TVINITIALIZATION', 0, ...
         'CONTINUATIONFACTOR', delta);
     
mse = norm(x-x_salsa,'fro')^2 /numel(x);
cpu_time = times(end);

calls_csalsa = calls;
calls = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure, image(x)
map = colormap(hsv);
map(1,:) = [1,1,1];
colormap(map), axis equal, colorbar;
title('Original');

figure, image(x_salsa),
colormap(map), axis equal, colorbar;
title('SALSA');


figure, image(Xnesta),
colormap(map), axis equal, colorbar;
title('NESTA');

fprintf('NESTA\nNumber of calls to A and AT: %d\n', calls_nesta)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g\n', niter, times_nesta(end), mse_nesta)

fprintf('C-SALSA\nNumber of calls to A and AT: %d\n', calls_csalsa)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g\n', length(times), times(end), mse)

figure, plot(times, objective, times_nesta, outData(:,2), 'r-.', 'Linewidth', 2.7), 
title('TV','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('C-SALSA', 'NESTA'),
xlabel('seconds'), 


figure, semilogy(times, mses, times_nesta, outData(:,1), 'r-.', 'Linewidth', 2.7), 
title('MSE','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('C-SALSA', 'NESTA'),
xlabel('seconds'), 

epsbar = epsilon*ones(1,10);
t_eps = linspace(0,max(times(end), times_nesta(end)),10);
figure, semilogy(t_eps, epsbar, 'ks', times, criterion, times_nesta, resid(:,1), 'r:', 'Linewidth', 3.5), 
title('Constraint violation','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('\epsilon', 'C-SALSA', 'NESTA', 'Location', 'SouthEast'),
xlabel('seconds'), 
