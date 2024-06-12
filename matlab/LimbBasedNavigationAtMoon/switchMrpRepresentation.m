function [mrp_s, flag] = switchMrpRepresentation(mrp)
    
    % [mrp_s, flag] = switchMrpRepresentation(mrp)
    %
    % A function computing the shadow set ot the Modified Rodriguez Parameter
    % from the Modified Rodriguez Parameter
    %
    % mrp: the Modified Rodriguez Parameter
    %
    % mrp_s: the shadow MRP
    % flag: 0 if the set has not switched, 1 othewise.
    %
    % Author: Paolo Panicucci
    
    flag = 0;
    s = dot(mrp,mrp);
    if(s>=1)
        mrp_s = -mrp/s;
        flag = 1;
    else 
        mrp_s = mrp;
    end
end