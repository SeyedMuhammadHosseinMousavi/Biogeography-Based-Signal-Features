%% Biogeography Based Optimization (BBO) Signal Features - Created in 19 Jan 2022 by Seyed Muhammad Hossein Mousavi
% This code employs Biogeography Based Optimization BBO evolutionary 
% algorithm in order to extract audio signal feature.
% System takes audio signal from input and extracts 'Energy Entropy'
% feature from spatial domain and 'Spectral Centroid' feature from
% frequency domain as input and target is 'Short Time Energy' feature from time
% domain. BBO algorithm fits the final feature vector based on its nature
% and returns evolutionary signal feature. BBO is suitable for this
% purpose due to its high speed for multiple signal feature extraction
% task. You can use your own audio signal or any other type of signal and play
% with parameters depending on your system and the lenght of the signal.
% ------------------------------------------------ 
% Feel free to contact me if you find any problem using the code: 
% Author: SeyedMuhammadHosseinMousavi
% My Email: mosavi.a.i.buali@gmail.com 
% My Google Scholar: https://scholar.google.com/citations?user=PtvQvAQAAAAJ&hl=en 
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi?tab=repositories 
% My ORCID: https://orcid.org/0000-0001-6906-2152 
% My Scopus: https://www.scopus.com/authid/detail.uri?authorId=57193122985 
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916#
% my RG: https://www.researchgate.net/profile/Seyed-Mousavi-17
% ------------------------------------------------ 
% Hope it help you, enjoy the code and wish me luck :)

%% Making Things Ready
clc;
clear;
warning('off');

%% Music Signal Data Load
[signal,fs] = audioread('tar.wav');
win = 0.050;
step = 0.050;
fs=44100;

%% Time Domain Features
EnergyEntropy = Energy_Entropy_Block(signal, win*fs, step*fs, 10)';
ShortTimeEnergy = ShortTimeEnergy(signal, win*fs, step*fs);

%% Frequency Domain Features
SpectralCentroid = SpectralCentroid(signal, win*fs, step*fs, fs);

%% Making Inputs and Targets
Inputs=[EnergyEntropy SpectralCentroid]';
Targets=ShortTimeEnergy';
data.Inputs=Inputs;
data.Targets=Targets;
data=JustLoad(data);

%% Generate Basic Fuzzy Model
ClusNum=3; % FCM Cluster Number
fis=GenerateFuzzy(data,ClusNum);

%% BBO Algorithm Learning
BBOfis=bboFCN(fis,data);   

%% BBO Results 
% BBO Feature Output
TrTar=data.TrainTargets;
TrainOutputs=evalfis(data.TrainInputs,BBOfis);
% Basic and BBO Features
BasicFeature=data.TrainTargets;
BBOFeature=TrainOutputs;
% BBO Train Error
Errors=data.TrainTargets-TrainOutputs;
MSE=mean(Errors.^2);
RMSE=sqrt(MSE);  
error_mean=mean(Errors);
error_std=std(Errors);
%% Plots
% Plot Features
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,1,1);
plot(data.TrainTargets,'-',...
'LineWidth',2,...
'MarkerSize',3,...
'MarkerEdgeColor','r',...
'Color',[0.9,0.1,0.1]);
hold on;
plot(TrainOutputs,'-',...
'LineWidth',2,...
'MarkerSize',3,...
'MarkerEdgeColor','r',...
'Color',[0.1,0.1,0.1]);
legend('Basic Feature','BBO Feature');
title('BBO Signal Feature');
xlabel('Sample Index');
grid on;
% Plot Error
subplot(3,1,2);
plot(Errors,':',...
'LineWidth',2,...
'MarkerSize',3,...
'MarkerEdgeColor','g',...
'Color',[0.1,0.7,0.1]);
legend('BBO Training Error');
title(['BBO Train Error (RMSE) Is =     ' num2str(RMSE)]);
grid on;
% Plot Distribution Fit Histogram
subplot(3,1,3);
h=histfit(Errors, 40);
h(1).FaceColor = [.8 .2 0.9];
title([' BBO Train Error (STD) Is =   ' num2str(error_std)]);
% Feature Spectrograms
figure;
subplot(2,1,1)
spectrogram(BasicFeature);title('Basic Feature');
colormap bone
subplot(2,1,2)
spectrogram(BBOFeature);title('BBO Feature');
