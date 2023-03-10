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
person_locs = [person_1_loc;person_2_loc;person_3_loc];
% yaw, pitch,roll for voice directivity
[p1y,p1p] = yaw_pitch(person_1_loc, array_loc);
[p2y,p2p] = yaw_pitch(person_2_loc, array_loc);
[p3y,p3p] = yaw_pitch(person_3_loc, array_loc);
p1ypr = [p1y,p1p,0];
p2ypr = [p2y,p2p,0];
p3ypr = [p3y,p3p,0];


Sources = AddSource('Location',person_1_loc, 'orientation', p1ypr,'Type','cardioid');
Sources = AddSource(Sources,'Location',person_2_loc, 'orientation', p2ypr,'Type','cardioid');
Sources = AddSource(Sources,'Location',person_3_loc, 'orientation', p3ypr,'Type','cardioid');




%% Simulation


[samples] = RunMCRoomSim(Sources,Receivers,Room,Options);

%% Saving data
for i = 1:7
    for j = 1:3
        fname = ['IRs\speaker_' num2str(j) '_to_mic_' num2str(i) '.mat']
        buffer = cell2mat(samples(i,j));
        save(fname,'buffer');
    end
end
save('IR_cells.mat','samples');
for i = 1:7
    fname = ['outputs\mic_' num2str(i) '_IR.txt']
    file = fopen(fname,'wt');
    IR(:,i) = cell2mat(samples(i,1)) + cell2mat(samples(i,2)) + cell2mat(samples(i,3));
    fprintf(file,'%f\n',IR(:,i));
end
fclose('all');
fname = 'outputs\theta.txt';
file = fopen(fname,'wt');
for i = 1:3
    theta(i) = rad2deg(atan2((person_locs(i,2) - array_loc(2)),(person_locs(i,1) - array_loc(1))));
end
fprintf(file,'%f\n',theta);



%% sounding
% soundsc(RIR,44100)





%% Visualization
figure

plot3(person_1_loc(1),person_1_loc(2),person_1_loc(3),'o');
hold on
plot3(person_2_loc(1),person_2_loc(2),person_2_loc(3),'o');
plot3(person_3_loc(1),person_3_loc(2),person_3_loc(3),'o');
plot3(mic_x,mic_y,mic_z, 'o');
text(mic_x,mic_y,mic_z,{'1','2','3','4','5','6','7'})
xlim([0 9]);
ylim([0 9]);
zlim([0 3]);
[x y] = meshgrid(0:0.1:9); % Generate x and y data
z = zeros(size(x, 1)); % Generate z data
plot3(x, y, z,'Color', 'black') % Plot the surface

xlabel('x');
ylabel('y');
zlabel('z');


%% Functions

% Generates arrays for the x, y, and z locations of mics in array
% 3 cm apart
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

% Gives yaw and pitch from one point to another
function [yaw,pitch] = yaw_pitch(speaker,mic_array)
    dx = speaker(1) - mic_array(1);
    dy = speaker(2) - mic_array(2);
    dz = speaker(3) - mic_array(3);
    yaw = atan(dy/dx);
    pitch = atan(sqrt(dy^2 + dx^2)/dz) + pi;



end