clear
clc
close all

flag_gif = false;

% 
addpath(genpath('dataset\preparation'))
addpath(genpath('src'))

%% HIL CAMERA
hil.N = 100; % random space scaling
hil.seed = 1;
hil.umax = 1440; % tinyv3rse screen
hil.vmax = 1440; 
hil.umid = hil.umax/2 + 0.5;
hil.vmid = hil.vmax/2 + 0.5;
hil.Rmin = 20; % Minimum disk size [px]
%hil.Rmin = 50.7e-3/18e-6*atan(atan(1.747e3/85e3)); % f/muPixel*atan(atan(RMoon/d_cam2moon))
hil.Rmax = 150; % Minimum disk size [px]
%hil.Rmin = 50.7e-3/18e-6*atan(atan(1.747e3/38.5e3)); % f/muPixel*atan(atan(RMoon/d_cam2moon))
hil.f = 180e-3;
hil.mu = 44.1e-6;

%% GT CAMERA
gt.umax = 1024;
gt.vmax = 1024;
gt.umid = gt.umax/2 + 0.5;
gt.vmid = gt.vmax/2 + 0.5;
gt.f = 50.7e-3;
gt.mu = 18e-6;

%% SAMPLE SPACE
[uC_hil, vC_hil, R_hil, alpha_hil, beta_hil] = sample_disks_params(hil.N, hil.seed, hil.umax, hil.vmax, hil.Rmin, hil.Rmax);

%% GENERATE DISKS

Np = 1e3;
x_hil = zeros(hil.N, Np);
y_hil = zeros(hil.N, Np);
u_hil = zeros(hil.N, Np);
v_hil = zeros(hil.N, Np);

for ix = 1:hil.N

    % Generate half disk (left) plus half ellipse (right)
    th = linspace(-pi/2, 3/2*pi, Np);
    x_hil(ix, th >= -pi/2 & th < pi/2) = cos(alpha_hil(ix)) * R_hil(ix) * cos(th(th >= -pi/2 & th < pi/2));
    x_hil(ix, th > pi/2 & th <= 3*pi/2) = R_hil(ix) * cos(th(th > pi/2 & th <= 3*pi/2));
    y_hil(ix, :) = R_hil(ix) * sin(th);

    % Apply roto-translation
    dcm = [-cos(beta_hil(ix)) sin(beta_hil(ix));
           sin(beta_hil(ix)) cos(beta_hil(ix))];
    uv_hil = [uC_hil(ix); vC_hil(ix)] + dcm*[x_hil(ix, :); y_hil(ix, :)];
    u_hil(ix, :) = uv_hil(1, :);
    v_hil(ix, :) = uv_hil(2, :);
end

fun_hil2gt = @(x, fmuhil, fmugt, xmidhil, xmidgt) fmugt/fmuhil*(x - xmidhil) + xmidgt;

% u_gt = mag.*u_hil;
% v_gt = mag.*v_hil;
% uC_gt = mag.*uC_hil;
% vC_gt = mag.*vC_hil;
% R_gt = mag.*R_hil;

u_gt = fun_hil2gt(u_hil, hil.f/hil.mu, gt.f/gt.mu, hil.umid, gt.umid);
v_gt = fun_hil2gt(v_hil, hil.f/hil.mu, gt.f/gt.mu, hil.vmid, gt.vmid);
uC_gt = fun_hil2gt(uC_hil, hil.f/hil.mu, gt.f/gt.mu, hil.umid, gt.umid);
vC_gt = fun_hil2gt(vC_hil, hil.f/hil.mu, gt.f/gt.mu, hil.vmid, gt.vmid);
R_gt = fun_hil2gt(R_hil, hil.f/hil.mu, gt.f/gt.mu, 0, 0);
alpha_gt = alpha_hil;
beta_gt = beta_hil;

%% CREATE HIL IMG

hil_dir = 'dataset\preparation\disk\hil';
mkdir(hil_dir)
for ix = 1:hil.N
    img = poly2mask(u_hil(ix, :), v_hil(ix, :), hil.umax, hil.vmax);
    imgfilename = sprintf('%06d.png',ix);
    imwrite(img, fullfile(hil_dir,imgfilename))
end

%% CREATE GT IMG 
gt_dir = 'dataset\preparation\disk\gt';
mkdir(gt_dir)
for ix = 1:hil.N
    img = poly2mask(u_gt(ix, :), v_gt(ix, :), gt.umax, gt.vmax);
    imgfilename = sprintf('%06d.png',ix);
    imwrite(img, fullfile(gt_dir,imgfilename))
end

%% CREATE METADATA
hil.u = uC_hil;
hil.v = vC_hil;
hil.R = R_hil;
hil.alpha = alpha_hil;
hil.beta = beta_hil;

%% GT CAMERA
gt.u = uC_gt;
gt.v = vC_gt;
gt.R = R_gt;
gt.alpha = alpha_gt;
gt.beta = beta_gt;

%% YAML
yaml.WriteYaml(fullfile(gt_dir,'metadata.yml'), hil, 0)
yaml.WriteYaml(fullfile(hil_dir,'metadata.yml'), hil, 0)

%% PLOT
% figure()
% grid on, hold on, axis equal
% plot(in.cx + in.res_px/2*cos(0:pi/100:2*pi), in.cy + in.res_px/2*sin(0:pi/100:2*pi),'r')
% plot(in.cx, in.cy, 'kx', 'MarkerSize',100)
% for ix = 1:in.N
%     p = plot(in.cx + xC(ix,:), in.cx + yC(ix,:), 'g');
%     xlim([0, in.cx*2])
%     ylim([0, in.cy*2])
%     pause(0.2)
%     %delete(p)
% end

figure()
grid on, hold on, axis equal
plot(hil.umid + hil.umax/2*cos(0:pi/100:2*pi), hil.vmid + hil.vmax/2*sin(0:pi/100:2*pi),'r')
plot(hil.umid, hil.vmid, 'kx', 'MarkerSize',100)
if flag_gif
% Create gif
gif(fullfile(hil_dir,'..\disk_sampling.gif'),'DelayTime',0.5)
end
for ix = 1:hil.N
    p = plot(u_hil(ix,:), v_hil(ix,:));
    xlim([0, hil.umid*2])
    ylim([0, hil.vmid*2])
    pause(0.2)
    %delete(p)
    try
    gif
    catch 
    end
end
legend('Center','FOV','Disks')

%% Verify segmentation
figure()
if flag_gif
% Create gif
gif(fullfile(hil_dir,'..\disk_images.gif'),'DelayTime',0.5)
end
for ix = 1:hil.N
    imgfilename = sprintf('%06d.png',ix);
    img = imread(fullfile(hil_dir,imgfilename));
    i = imshow(img);
    hold on
    p = plot(u_hil(ix,1:100:end), v_hil(ix,1:100:end),'go');
    xlim([0, hil.umid*2])
    ylim([0, hil.vmid*2])
    pause(0.5)
    try
    gif
    catch 
    end
    delete(i)
    delete(p)
end

%% Verify segmentation
figure()
if flag_gif
% Create gif
gif(fullfile(gt_dir,'..\disk_images.gif'),'DelayTime',0.5)
end
for ix = 1:hil.N
    imgfilename = sprintf('%06d.png',ix);
    img = imread(fullfile(gt_dir,imgfilename));
    i = imshow(img);
    hold on
    p(1) = plot(u_gt(ix,1:100:end), v_gt(ix,1:100:end),'go');
    p(2) = quiver(uC_gt(ix), vC_gt(ix), - R_gt(ix)*cos(gt.beta(ix)), R_gt(ix)*sin(gt.beta(ix)),'r','MarkerSize',3);
    xlim([0, gt.umid*2])
    ylim([0, gt.vmid*2])
    pause(1.5)
    try
    gif
    catch 
    end
    delete(i)
    delete(p)
end
