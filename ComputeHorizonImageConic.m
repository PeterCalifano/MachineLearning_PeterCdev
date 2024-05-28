function [o_dlimbPixCoords, o_dtightConeLocusImageMatrix] = ComputeHorizonImageConic(i_dKcam, ...
                                                                                     i_dShapeMatrix_TF, ...
                                                                                     i_dDCM_fromTFtoCAM, ...
                                                                                     i_dBodyPosVec_CAM, ...
                                                                                     i_dRangeOfAngles, ...
                                                                                     i_ui16Npoints)%#codegen
arguments
    i_dKcam
    i_dShapeMatrix_TF
    i_dDCM_fromTFtoCAM
    i_dBodyPosVec_CAM
    i_dRangeOfAngles
    i_ui16Npoints
end
%% PROTOTYPE
% [o_dlimbPixCoords, o_dtightConeLocusImageMatrix] = ComputeHorizonImageConic(i_dKcam, ...
%                                                                                      i_dShapeMatrix_TF, ...
%                                                                                      i_dDCM_fromTFtoCAM, ...
%                                                                                      i_dBodyPosVec_CAM, ...
%                                                                                      i_dRangeOfAngles, ...
%                                                                                      i_ui16Npoints)%#codegen
% -------------------------------------------------------------------------------------------------------------
%% DESCRIPTION
% What the function does
% REFERENCE:
% [1] J. A. Christian, “A Tutorial on Horizon-Based Optical Navigation and Attitude Determination With Space 
% Imaging Systems,” IEEE Access, vol. 9, pp. 19819–19853, 2021, doi: 10.1109/ACCESS.2021.3051914.
% -------------------------------------------------------------------------------------------------------------
%% INPUT
% in1 [dim] description
% Name1                     []
% Name2                     []
% Name3                     []
% Name4                     []
% Name5                     []
% Name6                     []
% -------------------------------------------------------------------------------------------------------------
%% OUTPUT
% out1 [dim] description
% Name1                     []
% Name2                     []
% Name3                     []
% Name4                     []
% Name5                     []
% Name6                     []
% -------------------------------------------------------------------------------------------------------------
%% CHANGELOG
% 25-05-2024        Pietro Califano         First version coded from ref. [1]
% -------------------------------------------------------------------------------------------------------------
%% DEPENDENCIES
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Future upgrades
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Function code

% Compute Body Tightly-bounding conic equation in Pixel coordinates
o_dtightConeLocusImageMatrix = ComputeTightConeLocusInImg(i_dKcam, i_dShapeMatrix_TF, i_dDCM_fromTFtoCAM, i_dBodyPosVec_CAM);

% Extract coefficients from the matrix (from GPT 4o)
A = o_dtightConeLocusImageMatrix(1,1); % A must be > 0
B = 2 * o_dtightConeLocusImageMatrix(1,2);
C = o_dtightConeLocusImageMatrix(2,2); % C must be > 0
D = 2 * o_dtightConeLocusImageMatrix(1,3);
E = 2 * o_dtightConeLocusImageMatrix(2,3);
F = o_dtightConeLocusImageMatrix(3,3);

assert(A>=0, "A coefficient must be >= 0")
assert(C>=0, "C coefficient must be >= 0")

% Calculate the center of the ellipse
delta = B^2 - 4*A*C;
ellipseCx = (2*C*D - B*E) / delta;
ellipseCy = (2*A*E - B*D) / delta;

% Calculate the angle of rotation
majorAxisAngleFromX = 0.5 * atan2(B, A - C);

% Calculate the semi-major and semi-minor axes
up = 2*(A*E^2 + C*D^2 + F*B^2 - B*D*E - A*C*F); % ??
down1 = (B^2 - 4*A*C) * ((C - A) + sqrt((A - C)^2 + B^2)); % ??
down2 = (B^2 - 4*A*C) * ((A - C) + sqrt((A - C)^2 + B^2)); % ??
semiMajorAx = sqrt(up / down1);
semiMinorAx = sqrt(up / down2);

% Parametric equation of the ellipse
if i_ui16Npoints == 0 && length(i_dRangeOfAngles) > 2
    pixAnglesToGet = i_dRangeOfAngles;
elseif i_ui16Npoints > 0 && length(i_dRangeOfAngles) == 2
    pixAnglesToGet = linspace(i_dRangeOfAngles(1), i_dRangeOfAngles(2), i_ui16Npoints);
else
    fprintf("\n Warning: incorrect inputs to get pixels! Using default value: [0, 2pi], 1000 points.")
    pixAnglesToGet = linspace(0, 2*pi, 1000);
end

% Compute pixels
o_dlimbPixCoords = coder.nullcopy(zeros(2, length(pixAnglesToGet)));

sinMajorAxFromX = sin(majorAxisAngleFromX);
cosMajorAxFromX = cos(majorAxisAngleFromX);

o_dlimbPixCoords(1, :) = ellipseCx + semiMajorAx*cos(pixAnglesToGet)*cosMajorAxFromX ...
    - semiMinorAx*sin(pixAnglesToGet)*sinMajorAxFromX;

o_dlimbPixCoords(1, :) = ellipseCy + semiMajorAx*cos(pixAnglesToGet)*sinMajorAxFromX ...
    + semiMinorAx*sin(pixAnglesToGet)*cosMajorAxFromX;

%% LOCAL FUNCTION
    function tightConeLocusImageMatrix = ComputeTightConeLocusInImg(i_dKcam, i_dShapeMatrix_TF, i_dDCM_fromTFtoCAM, i_dBodyPosVec_CAM)
        % ComputeTightConeLocusInImg = @(dShapeMatrix_CAM, dBodyPosVec_CAM) ...
        %                         dShapeMatrix_CAM * dBodyPosVec_CAM * dBodyPosVec_CAM' * dShapeMatrix_CAM ...
        %                         - (dBodyPosVec_CAM' * dShapeMatrix_CAM * dBodyPosVec_CAM - 1.0) * dShapeMatrix_CAM;

        dShapeMatrix_CAM = i_dDCM_fromTFtoCAM * i_dShapeMatrix_TF * i_dDCM_fromTFtoCAM';
        invKcam = eye(3)/i_dKcam;

        tightConeLocusImageMatrix = transpose(invKcam) *( (dShapeMatrix_CAM * i_dBodyPosVec_CAM) * (i_dBodyPosVec_CAM' * dShapeMatrix_CAM) ...
            - (i_dBodyPosVec_CAM' * dShapeMatrix_CAM * i_dBodyPosVec_CAM - 1.0) * dShapeMatrix_CAM) * invKcam;

        tightConeLocusImageMatrix(abs(tightConeLocusImageMatrix) < eps) = 0;

    end

end
