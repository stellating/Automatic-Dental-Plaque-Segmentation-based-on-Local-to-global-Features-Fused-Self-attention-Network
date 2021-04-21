close all;
clear all;
clc;


file_path =  '/home/lenovo/pzn/seg/git_ocnet/runs/datasets1104/mat_result_3_2284_val_20201104/';% 
new_file_path = '/home/lenovo/pzn/seg/git_ocnet/runs/datasets1104/cnn3_val/';

if(~exist(new_file_path,'dir'))
   mkdir(new_file_path); 
end

img_path_list = dir(strcat(file_path,'*.mat'));%
img_num = length(img_path_list);%

if img_num > 0 %
    for j = 1:img_num %
    %for j = 81:img_num %
    %for j = 1:2 %
        image_name = img_path_list(j).name(1:end-4);% 
        load(strcat(file_path,image_name,'.mat'));
        image_name % 
        Aimage =  feature3.pzn_feature_3;
        Aimage_i = Aimage(:,:,1);
        Bimage = imresize(Aimage,[700 700],'bilinear');      % resizeΪ256x256
        
        new_image_name = strcat(new_file_path,image_name,'.mat');
        save(new_image_name,'Bimage');
    end      
end
