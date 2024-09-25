% 
addpath(genpath('dataset\preparation'))
addpath(genpath('src'))

% Sample the raw image folders using a certain shift step.
path_raw = 'dataset\preparation\disk\htb\DISK30_0200_0100\raw';
filename_raw_prefix = 'DISK30_0200_0100';
Nshift = 5;
Nstep = 3;

postprocess_htb(path_raw, filename_raw_prefix, Nshift, Nstep)


