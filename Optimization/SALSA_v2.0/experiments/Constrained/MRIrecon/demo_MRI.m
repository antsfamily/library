close all;
clear all;

filename = 'demo_MRI';
addpath ./../../../src
addpath ./../../../utils
verbose = 0;

N = 128;

x = phantom(N);

angles = 22;

[mask_temp,Mh,mi,mhi] = LineMask(angles,N);
mask = fftshift(mask_temp);
OMEGA = mhi;

A = @(x)  masked_FFT(x,mask);
AT = @(x) (masked_FFT_t(x,mask));
ATA = @(x) (ifft2c(mask.*fft2c(x)));

B = @(z) masked_FFT(reshape(z,[N,N]),mask);
Bt = @(z) reshape(masked_FFT_t(z,mask), [N*N,1]);

sigma =  1e-3/sqrt(2);
y = A(x);
y = y + sigma*(randn(size(y)) + i*randn(size(y)));

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);
ATA = @(x) callcounter(ATA,x);

%%%% TV regularization
chambolleit = 10;
Psi_TV = @(x,th)  projk(real(x),th,chambolleit);
Phi_TV = @(x) TVnorm(real(x));

%%%% algorithm parameters
global counterA;
global counterAt;

stop_th = 1e-6;
mu1 = 1;
mu2 = mu1;
tau = mu1/mu2;
epsilon = sqrt(numel(y)+8*sqrt(numel(y)))*sigma;
iters = 500;

tau = mu1/mu2;
LS = @(x, mu1) (1/mu1)*( x - (1/(1+tau))*ATA(x) );

fprintf('Running C-SALSA...\n')
[z, numA, numAt, objective, distance1, distance2, criterion, times, mses] = ...
         csalsa_v2(y, A, mu1, mu2, sigma,...
         'AT', AT, ...
         'PHI', Phi_TV, ...
         'PSI', Psi_TV, ...
         'StopCriterion', 2, ...
         'True_x', x, ...
         'ToleranceA', stop_th,...
         'MAXITERA', iters, ...
         'LS', LS, ...
         'VERBOSE', 0, ...
         'EPSILON', epsilon, ...
         'INITIALIZATION', 2, ...
         'TVINITIALIZATION', 0, ...
         'TVITERS', 5, ...
         'CONTINUATIONFACTOR', 1.1);
mse = norm(x-z,'fro')^2 /numel(x);
cpu_time = times(end);

calls_csalsa = calls;
calls = 0;

%%%%%%%%% NESTA
n = N;
U = @(z) z;
Ut = @(z) z;
mu = 1e-6;
opts = [];
opts.maxintiter = 5;
opts.TOlVar = stop_th;
opts.verbose = 0;
opts.maxiter = 5000;
opts.U = U;
opts.Ut = Ut;
opts.stoptest = 1;  
opts.typemin = 'tv';
opts.outFcn = @(z) [norm(z-x(:),2)^2/numel(x),...
                    Phi_TV(reshape(z,[N,N]))];
delta = epsilon;
Ac = @(z) callcounter(B,z);
Atc = @(z) callcounter(Bt,z);

fprintf('Running NESTA...\n')
tic;
[x_nesta,niter,resid, outData, times_nesta] = NESTA(Ac,Atc,y,mu,delta,opts);
t.NESTA = toc;
times_nesta = monotonize(times_nesta);
tvnesta = calctv(n,n,reshape(x_nesta,n,n));
Xnesta = real(reshape(x_nesta,n,n));

mse_nesta = norm(Xnesta-x,'fro')^2/numel(x);

calls_nesta = calls;
calls = 0;


fprintf('C-SALSA\nNumber of calls to A and AT: %d\n', calls_csalsa)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g\n', length(times)-1, times(end), mse)

fprintf('NESTA\nNumber of calls to A and AT: %d\n', calls_nesta)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g\n', niter, t.NESTA, mse_nesta)


figure, imagesc(x), colormap gray, axis equal, axis off

figure, imagesc(mask), colormap gray, axis equal, axis off
title('Sampling Mask (22 beams)');

figure , imagesc(real(z)), colormap gray, axis equal, axis off;
title('Estimated using SALSA');

figure, plot(times, objective, times_nesta, outData(:,2), 'r:', 'Linewidth', 3.5), 
title('TV','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('C-SALSA', 'NESTA'),
xlabel('seconds'), 

figure, plot(times, mses, times_nesta, outData(:,1),'r:', 'Linewidth', 1.8), 
title('MSE','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('C-SALSA', 'NESTA'),
xlabel('seconds'), 

teps = linspace(0, max(times_nesta(end), times(end)), 10);
epsbar = epsilon*ones(size(teps));
figure, semilogy(teps, epsbar, 'ks', times, criterion, 'b', times_nesta, resid(:,1), 'r:', 'Linewidth', 3.5), 
title('Constraint violation','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('\epsilon', 'C-SALSA', 'NESTA'),
xlabel('seconds'), 

