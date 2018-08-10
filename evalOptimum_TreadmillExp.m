function a = evalOptimum_TreadmillExp(b, s, tHold, S1, S2)

b1 = b(1); b2 = b(2); b3 = b(3); b4 = b(4); b5 = b(5);

    if s < tHold
        % first speed perturb
        x = S1;
    else
        x = S2;
    end

    y = -(b2 + b4*x)/(2*b5);

    % optimal action
    a = [x y];
    
end