close all;
clear;

addpath ./../../../src
addpath ./../../../utils

% f = sparsePWS(256, 16, 12);
f = double( imread('orig12.tif') );
f = f./max(max(f));

[M, N] = size(f);

%%%% function handle for blur operator
L = 7;
h = ones(L,L);
h = h/sum(sum(h));

H = zeros(M,N);
H(1:L,1:L) = h;

H_FFT = fft2(H);
HC_FFT = conj(H_FFT);

A = @(x) real(ifft2(H_FFT.*fft2(x)));
AT = @(x) real(ifft2(HC_FFT.*fft2(x)));

%%%%% observation
g = A(f);
Psig  = norm(g,'fro')^2/(M*N);
BSNRdb = -1;
sigma = sqrt( Psig/( 10^(BSNRdb/10) ) );
g = g + sigma*randn(size(g));

%%%%% deblurring using twist algo, for l1 reg
% max no. of Bregman iterations
maxiter = 100;
% convergence threshold
tol = 1e-4;

linfnorm = max(max(abs(AT(g))));

% TV regularization parameter 
tau_TV = 0.05*linfnorm;

% l1 regularization parameter 
tau_l1 = (6e-2)*linfnorm;

% stopping theshold for IST/TwIST algo
tolA = 1e-6;

mu = tau_TV*5;

Phi_TV = @(x) TVnorm(x);
chambolleit = 20;
Psi_TV = @(x,th) projk(x,th,chambolleit); 


fprintf('Running SALSA with Compound Regularization\n')

Phi_l1 = @(x) sum(abs(x(:)));
Psi_l1 = @(x, lambda) soft(x, lambda);

y = g;
ATy = AT(y);

tau1 = tau_TV;
mu1 = tau1*5;

tau2 = tau_l1;
mu2 = tau2*5;

mu = mu1+mu2;
filter_FFT = 1./(abs(H_FFT).^2 + mu);
invLS = @(x) real( ifft2( filter_FFT.*fft2(x) ) );

[x_hat, numA, numAt, objective, distance, times, mse_compound] = CoRAL_v2(y,A,tau1,tau2,...
    'AT', AT, ...
    'MAXITERA', 200, ...
    'MU1', mu1, ...
    'MU2', mu2, ...
    'PHI1', Phi_TV, ...
    'PSI1', Psi_TV, ...
    'TVINITIALIZATION1', 1, ...
    'STOPCRITERION', 1, ...
    'TOLERANCEA', tol, ...
    'LS', invLS,...
    'TRUE_X', f, ...
    'VERBOSE', 0 );
ISNR_cr = 10*log10( mse_compound(1)./mse_compound );

fprintf('Compound Regularized\nIters = %d, CPU time = %3.3g seconds, Final objective = %3.3g, MSE = %3.3g, ISNR = %3.3g dB\n',...
    length(objective)-1, times(end), objective(end), mse_compound(end), ISNR_cr(end))

figure, imagesc(x_hat), colormap gray, axis equal, axis off
title('Estimated with Compound Regularization');

figure, plot(times, objective, 'LineWidth',3.5),
title('Objective function','FontName','Times','FontSize',20),
set(gca,'FontName','Times'),
set(gca,'FontSize',20),
xlabel('seconds');

figure, plot(times, mse_compound, 'LineWidth',3.5),
title('MSE','FontName','Times','FontSize',20),
set(gca,'FontName','Times'),
set(gca,'FontSize',20),
xlabel('seconds');
