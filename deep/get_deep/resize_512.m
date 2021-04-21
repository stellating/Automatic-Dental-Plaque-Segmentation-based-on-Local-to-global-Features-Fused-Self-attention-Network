close all;
clear all;
clc;


file_path =  '/home/lenovo/pzn/seg/git_ocnet/runs/datasets1104/mat_result_512_2284_20201104/';% 
new_file_path = '/media/lenovo/pzn_Files of Study/cnn512_train/';
%new_file_path = '/media/lenovo/pzn_Download Setup/cnn512_train/';
%new_file_path = '/home/lenovo/pzn/seg/git_ocnet/runs/datasets1104/cnn512_train/';
if(~exist(new_file_path,'dir'))
   mkdir(new_file_path); 
end

img_path_list = dir(strcat(file_path,'*.mat'));%
img_num = length(img_path_list);%

if img_num > 0 %
    for j = 2347:img_num %
%     for j = 1935:img_num %
    %for j = 1:1 %
        image_name = img_path_list(j).name(1:end-4);% 
        load(strcat(file_path,image_name,'.mat'));
        image_name % 
        Aimage =  feature512.pzn_feature_512;
        Aimage_i = Aimage(:,:,1);
        Bimage = imresize(Aimage,[700 700],'bilinear');      % resizeΪ256x256
        
        new_image_name = strcat(new_file_path,image_name,'.mat');
        save(new_image_name,'Bimage');
        
        clear feature512;
        clear Aimage;
        clear Bimage;
    end      
end
