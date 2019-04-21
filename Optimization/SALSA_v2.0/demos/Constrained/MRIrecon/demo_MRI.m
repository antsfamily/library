% Demo of MRI image reconstruction, with total variation. 
% 128*128 SHepp-Logan phantom, 22 radial lines.

close all;
clear;

addpath ./../../../src
addpath ./../../../utils

verbose = 0;

N = 128;
M = N;

x = phantom(N);

angles = 22;

[mask_temp,Mh,mi,mhi] = LineMask(angles,N);
mask = fftshift(mask_temp);
OMEGA = mhi;

A = @(x)  masked_FFT(x,mask);
AT = @(x) (masked_FFT_t(x,mask));
ATA = @(x) (ifft2c(mask.*fft2c(x))) ;

% B = @(z) masked_FFT(reshape(z,[N,N]),mask);
% Bt = @(z) reshape(masked_FFT_t(z,mask), [N*N,1]);

sigma =  1e-3/sqrt(2);
y = A(x);
y = y + sigma*(randn(size(y)) + i*randn(size(y)));


%%%% TV regularization
chambolleit = 10;
Psi_TV = @(x,th)  projk(real(x),th,chambolleit);
Phi_TV = @(x) TVnorm(real(x));

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);
ATA = @(x) callcounter(ATA,x);

%%%% algorithm parameters
stop_th = 1e-6;
mu1 = 1;
mu2 = mu1;
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
         'StopCriterion', 1, ...
         'True_x', x, ...
         'ToleranceA', stop_th,...
         'MAXITERA', iters, ...
         'LS', LS, ...
         'VERBOSE', 0, ...
         'EPSILON', epsilon, ...
         'INITIALIZATION', 2, ...
         'TVINITIALIZATION', 1, ...
         'TVITERS', 5, ...
         'CONTINUATIONFACTOR', 1.1);
mse = norm(x-z,'fro')^2 /numel(x);
cpu_time = times(end);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('C-SALSA\nNumber of calls to A and AT: %d\n', calls)
fprintf('Iters = %d, CPU time = %g seconds, MSE = %g\n', length(times)-1, times(end), mse)

figure, imagesc(x), colormap gray, axis equal, axis off,
title('Original phantom');

figure, imagesc(mask), colormap gray, axis equal, axis off
title('Sampling Mask (22 beams)');

figure , imagesc(real(z)), colormap gray, axis equal, axis off;
title('Estimated using SALSA');

figure, plot(times, objective, 'Linewidth', 1.8), 
title('TV','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

figure, plot(times, mses, 'Linewidth', 1.8), 
title('MSE','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
xlabel('seconds');

epsbar = epsilon*ones(size(times));

figure, plot(times, epsbar, 'r:', times, criterion, 'b', 'Linewidth', 2.4), 
title('Constraint violation','FontName','Times','FontSize',14),
set(gca,'FontName','Times'),
set(gca,'FontSize',14),
legend('\epsilon', 'C-SALSA'),
xlabel('seconds');

