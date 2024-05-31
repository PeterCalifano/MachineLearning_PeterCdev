clc;
clear;
close all;

%% LUMIO kernels and data loading script
RUN_RENDERING = false;

% programmatic_run = 1;
% addNoiseToImage = 0; % if 0 (adviced) will consider the same images for each Monte Carlo sample
% saveResults = 1;
runPreviousGeneratedSetting = 1;

% Find lumio-prototype home path

% lumio_path = cuborg.locate_home_path;
% kernelFolder = fullfile(lumio_path,'input','kernels','opnav');
% imageFolder = fullfile(lumio_path,'input','opnav_image_dataset', 'Images');
% saveFolder = fullfile(lumio_path,'output','mc_opnav');
lumio_path = fullfile('datasets', 'LUMIOdata');
kernelFolder = fullfile(lumio_path, "OpNavExperiment_LUMIO");
imageFolder = fullfile(lumio_path, 'ImagesWithError_LUMIO');
% saveFolder = fullfile(lumio_path,'output', 'mc_opnav');

% mkdir(saveFolder)
%
% if runPreviousGeneratedSetting
%     fileInitialSetting = '20240117T173152_MonteCarloSetting_1000Samples_3-6days_300s.mat';
%     ICsFileName = fileInitialSetting;
%     pathInitialSetting = fullfile(lumio_path, 'input', 'test_cases', 'opnav', fileInitialSetting);
%     load(pathInitialSetting)
% else
%     generateMCOpnavInputFile;
% end

% Load kernels
lumioSPK = fullfile(kernelFolder, 'Halo_Cj3p09_1yr.bsp');
lsk = fullfile(kernelFolder, 'naif0012.tls');
planetaryEph = fullfile(kernelFolder, 'de432s.bsp');
planetaryConstants = fullfile(kernelFolder, 'de-403-masses.tpc');

% Moon attitude kernels
LRO_KERNELS_PATH = char(fullfile("/media", "peterc", "Main", "Users/pietr/OneDrive - Politecnico di Milano", ...
    "PoliMi - LM","MATLABwideCodes/", "SPICE_KERNELS/", "LRO_kernels/"));

cspice_furnsh(strcat(LRO_KERNELS_PATH, 'moon_pa_de440_200625.bpc'));
cspice_furnsh(strcat(LRO_KERNELS_PATH, 'moon_de440_200625.tf'));
cspice_furnsh(strcat(LRO_KERNELS_PATH, 'lrorg_2023166_2023258_v01.bsp'));

cspice_furnsh(char(lumioSPK));
cspice_furnsh(char(lsk));
cspice_furnsh(char(planetaryEph));
cspice_furnsh(char(planetaryConstants));

% Initial and final UTC time of LUMIO's 1-year spk
utc0Traj = '2027-03-27 18:09:30';
utcEndTraj = '2028-03-21 12:00:00';

% Initial and final ephemeris time of the simulation
t0Sim = 0*86400;
et0 = cspice_str2et(utc0Traj) + t0Sim;
etEnd = et0 + 6*86400;
dtMeas = 300;
etVec = et0:dtMeas:etEnd; %linspace(et0, etEnd, 10000);
tSim = etEnd-et0;

% Recover inertial trajectories from spice
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

% Get Moon attitude at selected times
MoonFixedFrame = 'MOON_PA_DE440';
GlobalFrame = 'J2000';
R_Moon = 1737.4; % [km]

dDCM_Moon_fromTBtoIN = cspice_pxform(MoonFixedFrame, GlobalFrame, etVec);

% Position of spacecraft relative to Moon centred frame
xCamPosFromBody_IN = xTrueTrajIn(1:3,:) - MoonTrajIn(1:3,:);

% Position of the spacecraft relative to Sun
xCamPosFromSun_IN = xTrueTrajIn(1:3,:) - SunTrajIn(1:3,:);

% Construct LUMIO attitude matrices and relative attitude matrix (MoonSpacecraft)
% Per lumio, z verso la luna e y perpendicolare al sole
zAxisCam_IN = -xCamPosFromBody_IN./vecnorm(xCamPosFromBody_IN, 2, 1);
xAxisCam_IN = -xCamPosFromSun_IN./vecnorm(xCamPosFromSun_IN, 2, 1);
yAxisCam_IN = cross(zAxisCam_IN, xAxisCam_IN)./vecnorm(cross(zAxisCam_IN, xAxisCam_IN), 2, 1);

dDCM_fromINtoCAM = zeros(3, 3, size(xAxisCam_IN,2));
dDCM_fromTFtoCAM = zeros(3, 3, size(xAxisCam_IN,2));

for idT = 1:size(xAxisCam_IN, 2)
    dDCM_fromINtoCAM(:,:,idT) = transpose([xAxisCam_IN(:,idT), yAxisCam_IN(:,idT), zAxisCam_IN(:,idT)]);
    dDCM_fromTFtoCAM(:,:, idT) = dDCM_fromINtoCAM(:,:,idT) * transpose(dDCM_Moon_fromTBtoIN(:,:,idT));
end

i_dRbody = 1737.4;

if RUN_RENDERING == true

    %% Rendering
    % i_drCam
    % i_drTargetBody
    % i_drSun
    % o_dqSVRPminus_fromInvCAMtoIN
    % o_dqSVRPplus_fromINtoTBbl

    %% Camera definition
    CameraData.fov = 6; % [deg]
    CameraData.resx = 1024;
    CameraData.resy = 1024;

    %% Blender Options definition
    BlenderOpts.encoding = 8;
    BlenderOpts.rendSamples = 128;
    BlenderOpts.scene_viewSamples = 64;
    BlenderOpts.scattering = 0;

    %% Scenario data definition
    SceneData.scenarioName = "S6_Moon";
    SceneData.rStateCam    = i_drCam'; % In TF Blender frame
    SceneData.rTargetBody  = i_drTargetBody; % In TF Blender frame
    SceneData.rSun         = i_drSun'; % In TF frame
    % SceneData.qFromINtoCAM = o_dqSVRPminus_fromInvCAMtoIN; % Defined left handed
    % SceneData.qFromINtoTF  = o_dqSVRPplus_fromINtoTB'; % Left handed given as Right Handed "inverted"

    SceneData.qFromINtoCAM = o_dqSVRPminus_fromInvCAMtoIN'; % Defined left handed
    SceneData.qFromINtoTF  = o_dqSVRPplus_fromINtoTBbl'; % Left handed given as Right Handed "inverted"

    i_bVERBOSE_MODE = true;
    i_bCALL_BLENDER = true;
    i_bRUN_IN_BACKGROUND = true;

    % o_strConfigJSON NOT ASSIGNED INSIDE!
    % Allocate dimensional data as Reference: [km]
    % MoonDiamInBlender = 3474.84; % [km]

    drStateCam_IN = scaleBU2m * i_drCam;

    ReferenceData.drStateCam_IN = drStateCam_IN'; % [km]
    ReferenceData.dqSVRPplus_fromCAMtoIN = o_dqSVRPplus_fromCAMtoIN'; % Not in Blender convention (inverted Z)
    ReferenceData.dqSVRPplus_fromTBtoIN = o_dqSVRPplus_fromTBtoIN';
    % ReferenceData.dAzCAM_TF = AZ_Scatter; % [deg]
    % ReferenceData.dElCAM_TF = EL_Scatter; % [deg]
    ReferenceData.dSunDir_IN = (i_drSun./vecnorm(i_drSun', 2, 2)')'; % [deg]

    % Visual check before executing sequence of renderings

    [o_dSunPhaseAngle] = plot3DtrajectoryAndBoresight(drStateCam_IN, ...
        o_dqSVRPplus_fromCAMtoIN, i_drSun, i_dRbody);

    % Call to Blender API
    i_cPath2BlenderExe = '';

    if strcmpi(computer, 'PCWIN64')
        i_cPath2CORTO = 'C:\devDir\corto_PeterCdev';
    else
        i_cPath2CORTO = '/home/peterc/devDir/corto_PeterCdev';
    end

    input('----- PRESS ENTER TO START RENDERING -----')
    [o_strConfigJSON, o_ui8ImgSequence, o_cOutputPath] = MATLAB_BlenderCORTO_API( ...
        i_ui16Nposes,  ...
        SceneData, ...
        CameraData, ...
        BlenderOpts, ...
        ReferenceData, ...
        i_bCALL_BLENDER, ...
        i_cPath2CORTO, ...
        i_cPath2BlenderExe, ...
        i_bRUN_IN_BACKGROUND, ...
        i_bVERBOSE_MODE);

    if bSHOW_IMAGE_SEQUENCE == true
        % Check images
        figure;
        for id = 1:i_ui16Nimages
            imshow(o_ui8ImgSequence(:, :, id))
            pause(0.001)
        end
    end

    %% Generate video from image sequence
    % folderPath = fullfile(o_cOutputPath);

    imgSource = '/home/peterc/devDir/nav-backend/customExamples/matlab/CORTO_OUTPUT/S2_Itokawa_2024_05_23_19_09_09/img';

    fileName = 'ItokawaRender_23052024';
    imageSeq2Video(imgSource, fileName);

end

%% Test: labels generations
ERROR_VISUAL_CHECK = false;

imgID = 1;
imgDirStruct = getDirStruct(imageFolder);

idT = 1; % Due to loss of data in transferring images
imgName = imgDirStruct(imgID).name;
testImg = imread(fullfile(imageFolder, imgName));

% figure(1);
% imshow(testImg);

% Compute Tightly Bouding Cone (Apparent Horizon) matrix representation in Image plane
% ComputeTightConeLocusInImg = @(dShapeMatrix_CAM, dBodyPosVec_CAM) ...
%                         dShapeMatrix_CAM * dBodyPosVec_CAM * dBodyPosVec_CAM' * dShapeMatrix_CAM ...
%                         - (dBodyPosVec_CAM' * dShapeMatrix_CAM * dBodyPosVec_CAM - 1.0) * dShapeMatrix_CAM;


dShapeMatrix_TF = 1/R_Moon^2 .* eye(3);
dBodyPosVec_CAM = -dDCM_fromINtoCAM(:,:,idT)*xCamPosFromBody_IN(:,idT);
xCamPosFromBody_TF = dDCM_fromINtoCAM(:,:,idT) * transpose(dDCM_fromTFtoCAM(:,:,idT)) * xCamPosFromBody_IN(:,idT);

dKcam = zeros(3);
focalLength = 127; % [mm]
pixelSize = 13.3 * 1E-3; % [mm]

dKcam(1,1) = focalLength/pixelSize;
dKcam(2,2) = dKcam(1,1);
dKcam(3,3) = 1;
dKcam(1,3) = 1024/2;
dKcam(2,3) = 1024/2;

drangeCameraToBody = norm(xCamPosFromBody_IN(:,idT));

anglesRange = 0:0.1:2*pi;
displDirection = [cos(anglesRange); sin(anglesRange)];
dTargetPixAvgRadius = i_dRbody/drangeCameraToBody  * dKcam(1,1);

o_dSimExtractedLimbPixels(1:2, idW) = o_strConicData.dEllipseCentre + dTargetPixAvgRadius * displDirection;
saveStr.ui16coarseLimbPixels(:, idW) = uint16(mvnrnd(o_dSimExtractedLimbPixels(:, idW), sigmaPix));

% CODE FROM PAOLO:
% sx = 13.3e-3;
% sy = 13.3e-3;
% f = 127;
% cx =512+0.5;
% cy =512+0.5;
%
% Ktrue = [(f/sx) 0 cx; 0 (f/sy) cy; 0 0 1];


% Camera matrix
% P = K_c*[rotSC, tSC];

% Compute circle on the image by proejctin the sphere
% A_circle = inv(P*(A_sphere\P'));
% A_circle = A_circle./A_circle(3, 3)

tightConeLocusImageMatrix_PixCoords = ComputeTightConeLocusInImg(dKcam, dShapeMatrix_TF,...
    dDCM_fromTFtoCAM(:,:,idT), dBodyPosVec_CAM);

tightConeLocusImageMatrix_PixCoords = tightConeLocusImageMatrix_PixCoords./tightConeLocusImageMatrix_PixCoords(3,3);

[o_dtightConeLocusImageMatrix, o_strConicData, o_dlimbPixCoords] = ComputeHorizonImageConic(dKcam, ...
    dShapeMatrix_TF, ...
    dDCM_fromTFtoCAM(:,:,idT), ...
    dBodyPosVec_CAM, ...
    [0, pi/8, pi/4, pi/2, pi], ...
    0);


dHomogShapeMatrix_TF = diag([1, 1, 1, -R_Moon^2]);

[o_dConicInPixelCoord] = computeConic_directMethod(dKcam, ...
    dHomogShapeMatrix_TF, dDCM_fromTFtoCAM(:,:,idT),  dBodyPosVec_CAM);

diagEllipseMatrixInImgPix = eig(tightConeLocusImageMatrix_PixCoords);

% ACHTUNG: ellipse matrix is NOT positive definite... why?


%% ELLIPSE PLOT (by GPT4o)

% Extract coefficients from the matrix (from GPT)
A = tightConeLocusImageMatrix_PixCoords(1,1); % A must be > 0
B = 2 * tightConeLocusImageMatrix_PixCoords(1,2);
C = tightConeLocusImageMatrix_PixCoords(2,2); % C must be > 0
D = 2 * tightConeLocusImageMatrix_PixCoords(1,3);
E = 2 * tightConeLocusImageMatrix_PixCoords(2,3);
F = tightConeLocusImageMatrix_PixCoords(3,3);

if B == 0

    % Calculate the center of the ellipse
    x0 = -D / (2*A);
    y0 = -E / (2*C);

    numerator = 2 * (A*E^2 + C*D^2 - 4*A*C*F);
    denom1 = (B^2 - 4*A*C) * (A + C + abs(A - C));
    denom2 = (B^2 - 4*A*C) * (A + C - abs(A - C));

    a = sqrt(-numerator / denom1);
    b = sqrt(-numerator / denom2);

    theta = 0;

elseif B > 0

    % Calculate the center of the ellipse
    delta = B^2 - 4*A*C;
    x0 = (2*C*D - B*E) / delta; % Should be around image centre if Moon is near boresight
    y0 = (2*A*E - B*D) / delta;

    % Calculate the angle of rotation
    theta = 0.5 * atan2(B, A - C);

    % Calculate the semi-major and semi-minor axes
    up = 2*(A*E^2 + C*D^2 + F*B^2 - B*D*E - A*C*F);
    down1 = (B^2 - 4*A*C) * ((C - A) + sqrt((A - C)^2 + B^2));
    down2 = (B^2 - 4*A*C) * ((A - C) + sqrt((A - C)^2 + B^2));
    a = sqrt(up / down1);
    b = sqrt(up / down2);

end

% A = o_dConicInPixelCoord(1,1); % A must be > 0
% B = 2 * o_dConicInPixelCoord(1,2);
% C = o_dConicInPixelCoord(2,2); % C must be > 0
% D = 2 * o_dConicInPixelCoord(1,3);
% E = 2 * o_dConicInPixelCoord(2,3);
% F = o_dConicInPixelCoord(3,3);

% Parametric equation of the ellipse
t = linspace(0, 2*pi, 500);
X = x0 + a*cos(t)*cos(theta) - b*sin(t)*sin(theta);
Y = y0 + a*cos(t)*sin(theta) + b*sin(t)*cos(theta);

% Plot the ellipse
figure(1);
clf
imshow(testImg);
hold on
scatter(x0, y0, 8, 'g')
scatter(X, Y, 5, 'r','filled');
% axis equal;
xlabel('x');
ylabel('y');
title('Ellipse Plot');
grid on;

if ERROR_VISUAL_CHECK == true
    % Visualize error function from ellipse equation
    errFcn = @(u, v, invConicLocus) ([u, v, 1] * invConicLocus * [u; v; 1])^2;
    axisGrid = 0:1:1024;
    [Xgrid, Ygrid] = meshgrid(axisGrid);

    % errEval = errFcn(Xgrid, Ygrid, inv(tightConeLocusImageMatrix_PixCoords));
    errEval = zeros(size(Xgrid));
    % invConicLocus = inv(tightConeLocusImageMatrix_PixCoords);

    for idX = 1:length(axisGrid)
        for idY = 1:length(axisGrid)
            errEval(idX, idY) = errFcn(Xgrid(idX, idY), Ygrid(idX, idY), tightConeLocusImageMatrix_PixCoords);
        end
    end

    NumOfLevels = 1000;
    figure;
    contour3(Xgrid, Ygrid, errEval, NumOfLevels); % OK
    xlabel('X image axis')
    ylabel('Y image axis')
    zlabel('Z Error value')
    DefaultPlotOpts();
end

%% LOCAL FUNCTION
function tightConeLocusImageMatrix = ComputeTightConeLocusInImg(dKcam, dShapeMatrix_TF, dDCM_fromTFtoCAM, dBodyPosVec_CAM)
% ComputeTightConeLocusInImg = @(dShapeMatrix_CAM, dBodyPosVec_CAM) ...
%                         dShapeMatrix_CAM * dBodyPosVec_CAM * dBodyPosVec_CAM' * dShapeMatrix_CAM ...
%                         - (dBodyPosVec_CAM' * dShapeMatrix_CAM * dBodyPosVec_CAM - 1.0) * dShapeMatrix_CAM;

dShapeMatrix_CAM = dDCM_fromTFtoCAM * dShapeMatrix_TF * dDCM_fromTFtoCAM';
invKcam = eye(3)/dKcam;

tightConeLocusImageMatrix = transpose(invKcam) *( (dShapeMatrix_CAM * dBodyPosVec_CAM) * (dBodyPosVec_CAM' * dShapeMatrix_CAM) ...
    - (dBodyPosVec_CAM' * dShapeMatrix_CAM * dBodyPosVec_CAM - 1.0) * dShapeMatrix_CAM) * invKcam;

tightConeLocusImageMatrix(abs(tightConeLocusImageMatrix) < eps) = 0;

end

% Function to define LUMIO Attitude by Paolo
% function [A_BN, A_BN_hat] = AttitudeDefinition(rSCJ2000, rMJ2000, rSJ2000, sigmaAtt, randVec)
%
%
% rSCIn = rSCJ2000;
% rMIn = rMJ2000;
% rSIn = rSJ2000;
% mDirIn = -(rSCIn-rMIn)/norm(rSCIn-rMIn);
% sDirIn = -(rSCIn-rSIn)/norm(rSCIn-rSIn);
%
%
% zb = mDirIn;
% yb = cross(zb, sDirIn)/norm(cross(zb, sDirIn));
% xb = cross(yb, zb);
%
%
% A_BN = [xb'; yb'; zb'];
%
%
% eulErr = sigmaAtt.*randVec; %Attitude knowledge error - Euler angles
% Ae = eul2rotm(eulErr')';
% A_BN_hat = Ae*A_BN; %Estimated attitude
%
% end

function [o_dConicInPixelCoord] = computeConic_directMethod(i_dKcam, ...
    i_dHomogShapeMat_TF, i_dDCM_fromTFtoCAM, i_dxBodyFromCAM_CAM)

% Camera matrix
P_fromTFtoPix = i_dKcam*[i_dDCM_fromTFtoCAM, i_dxBodyFromCAM_CAM];

% i_dShapeMat_CAM = i_dDCM_fromTFtoCAM * i_dHomogShapeMat_TF * i_dDCM_fromTFtoCAM';

% Compute circle on the image by proejctin the sphere
o_dConicInPixelCoord = inv(P_fromTFtoPix*(i_dHomogShapeMat_TF\P_fromTFtoPix'));

o_dConicInPixelCoord = o_dConicInPixelCoord./o_dConicInPixelCoord(3, 3);

end
