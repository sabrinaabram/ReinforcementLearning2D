function a = evalOptimum(b, s, tHold, freqHold)

b1 = b(1); b2 = b(2); b3 = b(3); b4 = b(4); b5 = b(5);

    if s < tHold
        % from calcOptimum.m code
        y = freqHold;
    else
        y = (b1*b4 - 2*b2*b3)/(- b4^2 + 4*b3*b5);
    end

    % x min
    % sub in y min first
    x = -(b1 + b4*y)/(2*b3);

    % optimal action
    a = [x y];
    
end