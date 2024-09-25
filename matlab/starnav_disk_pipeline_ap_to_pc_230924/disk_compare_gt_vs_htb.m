% 
addpath(genpath('dataset\preparation'))
addpath(genpath('src'))

flag_gif = false;

path_gt = 'dataset\preparation\disk\gt';
filename_gt_prefix = '';
filename_gt_nzeros = 6;
Nshift_gt = 1;
Nstep_gt = 1;

path_htb = 'dataset\preparation\disk\htb\DISK30_0200_0100\png';
filename_htb_prefix = 'DISK30_0200_0100_';
filename_htb_nzeros = 6;
Nshift_htb = 5;
Nstep_htb = 3;

N = length({dir(path_htb).name});

figure('units','normalized','Position',[0.1 0.1 0.6 0.6])
subplot(1,2,1)
hold on
title('GT Binarized')
subplot(1,2,2)
hold on
title(['HIL ',filename_htb_prefix])

if flag_gif
    gif(fullfile(path_gt, '..\disk_comparison.gif'),'DelayTime',0.5);
end

for ix = 1:N
    
    filename_gt_suffix = sprintf(['%0',num2str(filename_gt_nzeros),'d.png'], Nshift_gt + (ix-1)*Nstep_gt);
    filepath_gt = fullfile(path_gt, [filename_gt_prefix, filename_gt_suffix]);
    
    filename_htb_suffix = sprintf(['%0',num2str(filename_htb_nzeros),'d.png'], Nshift_htb + (ix-1)*Nstep_htb);
    filepath_htb = fullfile(path_htb, [filename_htb_prefix, filename_htb_suffix]);

    if isfile(filepath_gt)
        img_gt = imread(filepath_gt);
        subplot(1,2,1)
        h_gt = imshow(img_gt);
    end
    if isfile(filepath_htb)
        img_htb = imread(filepath_htb);
        subplot(1,2,2)
        h_htb = imshow(img_htb);
    end

    if flag_gif
        gif;
    end

    pause(0.5)

    delete(h_gt)
    delete(h_htb)
end

%%

% Verify segmentation
figure('units','normalized','Position',[0.1 0.1 0.6 0.6])
hold on

if flag_gif
    gif(fullfile(path_gt, '..\disk_superimposition.gif'),'DelayTime',0.5)
end

for ix = 1:N
    
    filename_gt_suffix = sprintf(['%0',num2str(filename_gt_nzeros),'d.png'], Nshift_gt + (ix-1)*Nstep_gt);
    filepath_gt = fullfile(path_gt, [filename_gt_prefix, filename_gt_suffix]);
    
    filename_htb_suffix = sprintf(['%0',num2str(filename_htb_nzeros),'d.png'], Nshift_htb + (ix-1)*Nstep_htb);
    filepath_htb = fullfile(path_htb, [filename_htb_prefix, filename_htb_suffix]);

    if isfile(filepath_htb)
        img_htb = imread(filepath_htb);
        h_htb = imshow(img_htb);
        hold on
    end
    if isfile(filepath_gt)
        img_gt = imread(filepath_gt);
        img_gt_mask = repmat(double(img_gt), 1, 1, 3);
        img_gt_mask(:, :, [1, 3]) = 0;
        h_gt = imshow(img_gt_mask);
        h_gt.AlphaData = 0.5;
    end

    if flag_gif
        gif;
    end
    pause(0.5)

    delete(h_gt)
    delete(h_htb)
end