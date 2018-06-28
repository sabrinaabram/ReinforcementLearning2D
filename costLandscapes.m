function [b,natOptS,natOptF] = costLandscapes(n)

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
    S = round(100*S)/100; % this is nec because some values are not exactly at certain speeds
    
    % fit polynomial
    % p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2
    sfspfit = fit([S(:),F(:)],E(:),'p00 + p10*x + p01*y + p20*x^2 + p02*y^2');
    if 1
        plot(sfspfit,[S(:),F(:)],E(:))
    end

    % natural objective function
    b = [sfspfit.p00 sfspfit.p00;...
          sfspfit.p10 sfspfit.p01;...
          sfspfit.p20 sfspfit.p02];

    % find the energy optimum of the natural cost landscape
    [natOpt natOptidx] = min(E(:));
    natOptS = -sfspfit.p10/(2*sfspfit.p20);
    natOptF = -sfspfit.p01/(2*sfspfit.p02);

end