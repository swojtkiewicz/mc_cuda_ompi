%close all
clear variables
clc

%% display properties
set(0, 'DefaultAxesFontname', 'Arial CE');
font_size = 16;
set(0, 'DefaultAxesFontsize', font_size);
set(0, 'defaulttextfontname', 'Arial CE');
set(0, 'defaulttextfontsize', font_size);
set(0, 'defaultfigurecolor', 'w')

%% BODY

% fid= fopen('voxels_MPP_em_det_pair_2_opt_prop_1.vox','r');
% fid= fopen('voxels_MTSF_em_det_pair_2_opt_prop_1.vox','r');
% fid= fopen('voxels_VSF_em_det_pair_2_opt_prop_1.vox','r');
fid= fopen('voxels_out_1.vox','r');

x_dim = fread(fid,1,'uint32');
y_dim = fread(fid,1,'uint32');
z_dim = fread(fid,1,'uint32');
data = fread(fid,x_dim*y_dim*z_dim,'float32','l');
fclose(fid);

intensity = zeros(z_dim,x_dim);
   
N_photons = 50e6;
vox_size = 0.03;

% for ind_y = round(y_dim/2-1):round(y_dim/2-1)
for ind_y = 0:y_dim-1
    for ind_z = 0:z_dim-1
        for ind_x = 0:x_dim-1
            
%             if (ind_z < 302)
%                 data(ind_x + x_dim*ind_y + x_dim*y_dim*ind_z + 1) = 0;
%             end
            
            current_weight = data(ind_x + x_dim*ind_y + x_dim*y_dim*ind_z + 1);
            
            if current_weight ~= 0
                intensity(ind_z+1,ind_x+1) = intensity(ind_z+1,ind_x+1) + current_weight/N_photons/vox_size/vox_size;
            end
        end
    end
end

% 
% % save for paraview
% fid = fopen('voxels_paraview_head_t_3000ps.raw','w','l');
% data = log10(data/N_photons/vox_size/vox_size);
% data(~isfinite(data)) = NaN;%min(data(isfinite(data)));
% fwrite(fid,data,'float32','l');
% fclose(fid);

% clear data
%%

vox_size = 0.03;
% vox_size = 1;

intens = intensity;

% intens = intensity;
if ~any(intens < 0)
    intens = intensity/max(max(abs(intensity)));
    intens = log10(intens);
    % intens = intens - min(min(intens(intensity ~= 0)));
    intens(intens < -8) = -inf;
end




figure()
imagesc(linspace(0,x_dim*vox_size,x_dim),linspace(0,z_dim*vox_size,z_dim),flipud(intens))
% imagesc(linspace(0,x_dim*vox_size,x_dim),linspace(0,z_dim*vox_size,z_dim),flipud(log10(intensity/max(max(intensity)))))
%     imagesc(linspace(0,x_dim*vox_size,x_dim),linspace(0,y_dim*vox_size,y_dim),flipud(intens))
hold on
xlabel('y [mm]', 'interpreter','latex','FontSize',18)
ylabel('x [mm]', 'interpreter','latex','FontSize',18)
set(gca,'DataAspectRatio',[1 1 1]);
%     set(gca,'Clim',[-1 1])
hc = colorbar('location','EastOutside');
% set(get(hc,'Ylabel'),'String','mm','Interpreter','latex','FontSize',20);


return;






