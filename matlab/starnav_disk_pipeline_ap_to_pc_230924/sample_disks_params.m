function [u, v, R, alpha, beta] = sample_disks_params(N, seed, resx, resy, Rmin, Rmax)

%% DATA
cx = resx/2 + 0.5;
cy = resy/2 + 0.5;

%% SEED 
rng(seed)

%% SAMPLING

% Disk Radius
NR = N; 
R = Rmin + (Rmax - Rmin)*rand(NR, 1);

% Disk Position
Nr = N;
Nth = N;
rmax = min(resx/2 - Rmax, resy/2 - Rmax);
r = rmax.*sqrt(rand(Nr, 1));
th = 2*pi*rand(Nth, 1);
u = cx + r.*cos(th);
v = cy + r.*sin(th);

% Alpha angle
Na = N;
alpha = pi*rand(Na, 1);

% Sun direction
Nb = N;
beta = 2*pi*rand(Nb, 1);

%% PLOT
nh = 10;

figure()
subplot(2,3,1)
histogram(R, nh)
xlabel('R [px]')

subplot(2,3,2)
histogram(u, nh)
xlabel('u [px]')

subplot(2,3,3)
histogram(v, nh)
xlabel('v [px]')

subplot(2,3,4)
histogram(rad2deg(alpha), nh)
xlabel('\alpha [deg]')

subplot(2,3,5)
histogram(rad2deg(beta), nh)
xlabel('\beta [deg]')

%% 
kM = 60;
figure()
grid on, hold on, axis equal
scatter(u, v, kM*R, rad2deg(alpha))
plot(cx + resx/2*cos(0:pi/100:2*pi), cy + resy/2*sin(0:pi/100:2*pi),'r')
plot(cx, cy, 'kx', 'MarkerSize',100)
colormap(flip(colormap('turbo')))
c = colorbar;
c.Label.String = '\alpha [deg]';

end