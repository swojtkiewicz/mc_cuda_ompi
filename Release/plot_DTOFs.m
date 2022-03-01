close all
clear all
clc

%% display properties
set(0, 'DefaultAxesFontname', 'Arial CE');
font_size = 16;
set(0, 'DefaultAxesFontsize', font_size);
set(0, 'defaulttextfontname', 'Arial CE');
set(0, 'defaulttextfontsize', font_size);
set(0, 'defaultfigurecolor', 'w')

%% BODY

a = load('DTOFs_1.txt');
% a = load('DTOFs_em_det_pairs_0-1.5.txt');

no_of_pairs = size(a,2);
N = a(2,1)/2;
dt = a(1,1);
detectors = [a(3,:)' a(4,:)' a(5,:)' a(6,:)']; % [x y z r] in voxels
sources = [a(7,:)' a(8,:)' a(9,:)' a(10,:)']; % [x y z r] in voxels

refl = a(11:N+11-1,:);
fluo = a(N+11:end,:);
time = dt:dt:N*dt;

windowSize = 4/256*N;
% refl = filtfilt(ones(1,windowSize)/windowSize,1,refl);
% fluo = filtfilt(ones(1,windowSize)/windowSize,1,fluo);
% refl = abs(refl);
% fluo = abs(fluo);


refl_sum = mean(refl,2);
fluo_sum = mean(fluo,2);

dtof = [time' refl_sum];
dtoa = [time' fluo_sum];
% ind_max = find(refl_sum == max(refl_sum));

% moments = zeros(no_of_pairs,5);
% for ind = 1:no_of_pairs
%     moments(ind,:) = DTOF_stat_moments(dtof,5)';
% end


% figure()
% hold on
% for ind = 1:no_of_pairs
%     plot3(sources(ind,1),sources(ind,2),sources(ind,3),'xr')
%     plot3(detectors(ind,1),detectors(ind,2),detectors(ind,3),'ob')
%     plot3([sources(ind,1) detectors(ind,1)],[sources(ind,2) detectors(ind,2)],[sources(ind,3) detectors(ind,3)],'-k')
% end
% hold off

figure()
subplot(1,2,1)
semilogy(time,refl)
hold on
% semilogy([moments(2) moments(2)],[0.0001 max(refl_sum)],'k','LineWidth',2)
hold off

subplot(1,2,2)
semilogy(time,fluo)
hold on
% semilogy([moments(2) moments(2)],[0.0001 max(refl_sum)],'k','LineWidth',2)
hold off


% moments(2)
% moments(1)/50e6


