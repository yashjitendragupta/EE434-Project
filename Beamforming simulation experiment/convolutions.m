clear all
addpath('IRs\')
addpath('outputs\')
addpath('Input_voices\')
load("IR_cells.mat")
%% prep on files
N = length(cell2mat(samples(1,1)));
[voice_1, Fs] = audioread('Input_voices\voice_1.wav');
[voice_2, Fs] = audioread('Input_voices\voice_2.wav');
[voice_3, Fs] = audioread('Input_voices\voice_3.wav');
voices = [voice_1,voice_2,voice_3];
microphone = zeros(length(voice_1) + N,7);

%% convolutions
for i = 1:7
    for j = 1:3
        convolved{i,j} = ifft( ...
            fft([cell2mat(samples(i,j)); ...
                 zeros(length(voices(:,1)),1)]) ...
                 .* ...
                 fft([voices(:,j); ...
                 zeros(N,1)])); 
    end
end

%% combining

for i = 1:7
    microphone(:,i) = cell2mat(convolved(i,1)) + ... 
                      cell2mat(convolved(i,2)) + ...
                      cell2mat(convolved(i,3));
end

%% writing files

for i = 1:7
    audiowrite(['outputs\mic_' num2str(i) '_voices.wav'], microphone(:,i), Fs);
end


% soundsc(microphone(:,1),Fs)