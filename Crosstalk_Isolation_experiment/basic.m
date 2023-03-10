clear all
close all
%% Signal Loading
load('handel.mat')
A = y;
A = normalize_audio(A);
A = trim_audio(A);
load('gong.mat')
B = y;
B = normalize_audio(B);
B = trim_audio(B);
% A = resample(A,44100,Fs);
% B = resample(B,44100,Fs);
% Fs = 44100;
[A,B] = make_equal_length(A,B);
N = length(A);
time = (1:N)/Fs;
A = [time' A];
B = [time' B];

%% constants
% Alpha is magnitude reduction coefficient
% Beta is sample delay
% Speakers are A and B respectively, Microphones are C and D
% AC means between A and C, etc.
alpha_AC = 1;
alpha_AD = 1;
alpha_BC = 1;
alpha_BD = 1;
beta_AC = floor(.002*Fs);
beta_AD = floor(.001*Fs);
beta_BC = floor(.002*Fs);
beta_BD = floor(.004*Fs);

% Noisy Derived Constants
alpha_AC_d = alpha_AC;
alpha_AD_d = alpha_AD;
alpha_BC_d = alpha_BC;
alpha_BD_d = alpha_BD;
shift = 0;
beta_AC_d = beta_AC + ceil(shift*2*(rand-.5));
beta_AD_d = beta_AD + ceil(shift*2*(rand-.5));
beta_BC_d = beta_BC + ceil(shift*2*(rand-.5));
beta_BD_d = beta_BD + ceil(shift*2*(rand-.5));
%% Sim
sim('Signal_Mixer.slx','FixedStep',num2str(1/Fs));
C = ans.C;
D = ans.D;
% soundsc(C.data)
% soundsc(D.data);

%% Recovery

sim('Signal_Demixer.slx','FixedStep',num2str(1/Fs));
A_recov = ans.A_Recov;
B_recov = ans.B_Recov;
%% Postprocessing 
A_recov = trim_audio(A_recov);
B_recov = trim_audio(B_recov);

% soundsc(A_recov)
% soundsc(B_recov)

%% Audio Writing
audiowrite('A.wav',A(:,2),Fs)
audiowrite('B.wav',B(:,2),Fs)
audiowrite('C.wav',C.data/(max(abs(C.data))),Fs)
audiowrite('D.wav',D.data/(max(abs(D.data))),Fs)
audiowrite('A_recov.wav',normalize_audio(A_recov),Fs)
audiowrite('B_recov.wav',normalize_audio(B_recov),Fs)
%% Function Tests
close all
% slide((1:10)',3)
% sliding_similarity([0;0;0;(1:5)'],(1:5)')
% sliding_similarity((1:5)',(5:1)')
% A_error = sliding_similarity(A(:,2),A_recov)
[min,Nslide] = sliding_similarity(C.data,D.data)

% B_error = sliding_similarity(B(:,2),B_recov)
% recovered_error = sliding_similarity(A_recov,B_recov)
% original_error = sliding_similarity(A(:,2),B(:,2))
% A_D_error = sliding_similarity(A(:,2),D.data)
% C_D_error = sliding_similarity(C.data,D.data)
% A_B_recov = sliding_similarity(A(:,2),B_recov)
% B_A_recov = sliding_similarity(B(:,2),A_recov)




%% Aux Functions
% Trims 0 values at the start and end of array
function trimmed = trim_audio(A)
    first_val = find(A ~= 0, 1, 'first');
    last_val = find(A ~= 0, 1, 'last');
    trimmed = A(first_val:last_val);
end



function [similarity,N_shift] = sliding_similarity(A,B)
    [A,B] = make_equal_length(A,B);
    similarity_array = zeros(1,201);
    for i = (-100):(100)
        similarity_array(i+101) = immse(abs(A),abs(slide(B,i)));
    end
    figure
    normalize_audio(similarity_array);
    hold on
    plot(1-similarity_array)
%     plot(isoutlier(similarity_array));
    similarity = min(similarity_array);
    [dummy,N_shift] = max(1-similarity_array);
    N_shift = 101-N_shift;
end

% scales array so max value is 1
function normalized = normalize_audio(A)
    normalized = A/max(abs(A));
end

function out = slide(A,n)
    N = length(A);
    if(n<0)
        out = [A(-1*n+1:end); zeros(-1*n,1)];
    end
    if(n>0)
        out = [zeros(n,1); A(1:N-n)];
    end
    if(n==0)
        out = A;
    end
end

% Makes arrays equal length by end padding with zeros
function [a_ret,b_ret] = make_equal_length(A,B)

    NA = length(A);
    NB = length(B);
    if(NA > NB)
        B = [B;zeros(NA-NB,1)];     
    end
    if(NB > NA)
        A = [A;zeros(NB-NA,1)];
    end
    a_ret = A;
    b_ret = B;
end


