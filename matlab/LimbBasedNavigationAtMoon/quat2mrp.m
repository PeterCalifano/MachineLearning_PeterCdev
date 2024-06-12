function mrp = quat2mrp(q)

    % mrp = quat2mrp(q)
    %
    % A function computing the Modified Rodriguez Parameters given 
    % the quaternion
    %
    % q: the quaternion
    %
    % mrp: the Modified Rodriguez Parameters
    %
    % Author: Paolo Panicucci

    q_0 = q(1);
    
    if(abs(q_0+1)<1e-6)
       q = -q;
       q_0 = q(1);
    end
    
    mrp = switchMrpRepresentation(q(2:4)/(1+q_0));
end