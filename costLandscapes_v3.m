function [b,Spref,Fpref] = costLandscapes_v2(n)

    % speed frequency 2D relationship
    % Load Cost Contour Data
    % This data is generated in a separate m-file called CostContours.m. It uses
    % data that we collected in our own lab (Jess' JAP data) and combines it with
    % Umberger's step frequency data to approximate gross COT as a function of both
    % speed and step frequency. I then fit a model to that simulated data, and then
    % use the model to predict cost at gridded combinations of frequency and speed.
    load('CostContourData');
    % S: Speed grid from 0.75 - 1.75 m/s in 0.01 m/s increments (dif freqs are rows and dif speeds are columns) 
    % F: Step Freq grid from 66-147 spm in 1 spm increments (dif freqs are rows and dif speeds are columns) 
    % E: gross COT normalized for min value which occurs at 1.32 m/s and 107 spm in this smoothed data.
    % sp: a vector of all the speeds
    % fp: a vector of the preferred freq at each of the speeds. 
    % Note that speeds are rounded to the closest 0.01 and frequencies to the nearest 1 spm
    
    % APPROXIMATE absolute minimum
    b_abs = fit([S(:),F(:)],E(:),'poly22');
    % find the optimum in absolute units
    Fpref = (b_abs.p10*b_abs.p11 - 2*b_abs.p01*b_abs.p20)/(- b_abs.p11^2 + 4*b_abs.p20*b_abs.p02);
    Spref = -(b_abs.p10 + b_abs.p11*Fpref)/(2*b_abs.p20);
    
    % make units in % from preferred (or optimal)
    S = ((S - Spref)./Spref).*100;
    F = ((F - Fpref)./Fpref).*100;
    
    % new, more accurate fit in new units of % from preferred
    % x = speed, y = frequency
    b = fit([S(:),F(:)],E(:),'poly22');
    if 0
        plot(b,[S(:),F(:)],E(:))
    end
    
    % b.p00 just offset
    b = [b.p10 b.p01 b.p20 b.p11 b.p02]';
    
end