function convertAzm = azimuthPreference()
%MATLAB functions that return an azimuth use the convention clockwise from
%north, i.e. 0° north, east is 90°, west is 270°
%If you want to convert MATLAB azimuths to counter-clockwise from
%south, i.e. 0° south, + to east, - to west, then set convertAzm to true.
%If you want to use the MATLAB convertion, set convertAzm to false.

%comment out the option you don't want to use
%convertAzm = true; % convert azimuths to counter-clockwise from south
convertAzm = false; % leave azimuths as clockwise from north

end