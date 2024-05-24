clc;
clear;
close all;
programmatic_run = 1;
addNoiseToImage = 0; % if 0 (adviced) will consider the same images for each Monte Carlo sample
saveResults = 1;
runPreviousGeneratedSetting = 1;
% Find lumio-prototype home path
lumio_path = cuborg.locate_home_path;
kernelFolder = fullfile(lumio_path,'input','kernels','opnav');
imageFolder = fullfile(lumio_path,'input','opnav_image_dataset', 'Images');
saveFolder = fullfile(lumio_path,'output','mc_opnav');
mkdir(saveFolder)
if runPreviousGeneratedSetting
    fileInitialSetting = '20240117T173152_MonteCarloSetting_1000Samples_3-6days_300s.mat';
    ICsFileName = fileInitialSetting;
    pathInitialSetting = fullfile(lumio_path, 'input', 'test_cases', 'opnav', fileInitialSetting);
    load(pathInitialSetting)
else
    generateMCOpnavInputFile;
end
lumioSPK = fullfile(kernelFolder, 'Halo_Cj3p09_1yr.bsp');
lsk = fullfile(kernelFolder, 'naif0012.tls');
planetaryEph = fullfile(kernelFolder, 'de432s.bsp');
planetaryConstants = fullfile(kernelFolder, 'de-403-masses.tpc');
cspice_furnsh({lumioSPK, lsk, planetaryEph, planetaryConstants});
%Initial and final UTC time of LUMIO's 1-year spk
utc0Traj = '2027-03-27 18:09:30';
utcEndTraj = '2028-03-21 12:00:00';
%Initial and final ephemeris time of the simulation
t0Sim = 0*86400;
et0 = cspice_str2et(utc0Traj) + t0Sim;
etEnd = et0 + 6*86400;
dtMeas = 300;
etVec = et0:dtMeas:etEnd; %linspace(et0, etEnd, 10000);
tSim = etEnd-et0;
%Recover inertial trajectories from spice
xTrueTrajIn = cspice_spkezr('-100009', etVec, 'J2000', 'NONE', 'EARTH MOON BARYCENTER');
MoonTrajIn = cspice_spkezr('MOON', etVec, 'J2000', 'NONE', 'EARTH MOON BARYCENTER');
EarthTrajIn = cspice_spkezr('EARTH', etVec, 'J2000', 'NONE', 'EARTH MOON BARYCENTER');
SunTrajIn = cspice_spkezr('SUN', etVec, 'J2000', 'NONE', 'EARTH MOON BARYCENTER');
etVecTimeSeries = etVec-etVec(1);
xTrueTrajInTime = [etVecTimeSeries; xTrueTrajIn];
xTrueTrajInTime = xTrueTrajInTime';
MoonTrajInTime = [etVecTimeSeries; MoonTrajIn(1:3, :)];
MoonTrajInTime = MoonTrajInTime';
EarthTrajInTime = [etVecTimeSeries; EarthTrajIn(1:3, :)];
EarthTrajInTime = EarthTrajInTime';
SunTrajInTime = [etVecTimeSeries; SunTrajIn(1:3, :)];
SunTrajInTime = SunTrajInTime';
