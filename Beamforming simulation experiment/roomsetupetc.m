close all
clear all
addpath('MCRoomSim')

%% Room setup
Absorption_freqs = 	[125, 250, 500, 1000, 2000, 4000];
Absorption_coeffs = [0.14,0.1,0.06,0.05,0.04,0.04;
                     0.14,0.1,0.06,0.05,0.04,0.04;
                     0.14,0.1,0.06,0.05,0.04,0.04;
                     0.14,0.1,0.06,0.05,0.04,0.04;
                     0.01,0.02,0.06,0.15,0.25,0.45;
                     0.15,0.11,0.04,0.04,0.07,0.08];
Room = SetupRoom('Dim', [9,6,3],'Freq',Absorption_freqs,'Absorption', Absorption_coeffs);
Options = MCRoomSimOptions('Fs',44100,'AutoCrop',false)


%% Array Setup
array_loc = [4.5,3.5,.7];
[mic_x,mic_y,mic_z] = Arraygen(array_loc);
Receivers = AddReceiver('Location', [mic_x(1),mic_y(1),mic_z(1)]);
for i = 2:7
    Receivers = AddReceiver(Receivers,'Location', [mic_x(i),mic_y(i),mic_z(i)]);
end



%% Person setup
% [x,y,z]
person_1_loc = [3,2,1];
person_2_loc = [7,3,1];
person_3_loc = [3,5,1];
% yaw, pitch,roll
[p1y,p1p] = yaw_pitch(person_1_loc, array_loc);
[p2y,p2p] = yaw_pitch(person_2_loc, array_loc);
[p3y,p3p] = yaw_pitch(person_3_loc, array_loc);
p1ypr = [p1y,p1p,0];
p2ypr = [p2y,p2p,0];
p3ypr = [p3y,p3p,0];
% p1vec = [cos(p1y), sin(p1y), cos(p1p)];
% p1vec = p1vec + person_1_loc;

Sources = AddSource('Location',person_1_loc, 'orientation', p1ypr,'Type','cardioid');
Sources = AddSource(Sources,'Location',person_2_loc, 'orientation', p2ypr,'Type','cardioid');
Sources = AddSource(Sources,'Location',person_3_loc, 'orientation', p3ypr,'Type','cardioid');




%% Simulation


% [samples] = RunMCRoomSim(Sources,Receivers,Room,Options);
% audiowrite('SampleIR.wav',RIR,44100)


%% sounding
% soundsc(RIR,44100)





%% Visualization
figure

plot3(person_1_loc(1),person_1_loc(2),person_1_loc(3),'o');
hold on
plot3(person_2_loc(1),person_2_loc(2),person_2_loc(3),'o');
plot3(person_3_loc(1),person_3_loc(2),person_3_loc(3),'o');
plot3(mic_x,mic_y,mic_z, 'o');
xlim([0 9]);
ylim([0 6]);
zlim([0 3]);
[x y] = meshgrid(0:0.1:9); % Generate x and y data
z = zeros(size(x, 1)); % Generate z data
plot3(x, y, z,'Color', 'black') % Plot the surface


%% Functions

% Generates arrays for the x, y, and z locations of mics in array
function [x,y,z] = Arraygen(origin)
    x = origin(1);
    y = origin(2);
    z = origin(3);
    for i = 0:5
        x = [x origin(1)+.03*cos(pi*i/3)];
        y = [y origin(2)+.03*sin(pi*i/3)];
        z = [z origin(3)];
    end
end

function [yaw,pitch] = yaw_pitch(speaker,mic_array)
    dx = speaker(1) - mic_array(1);
    dy = speaker(2) - mic_array(2);
    dz = speaker(3) - mic_array(3);
    yaw = atan(dy/dx);
    pitch = atan(sqrt(dy^2 + dx^2)/dz) + pi;



end