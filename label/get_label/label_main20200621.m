close all;
clear all;
clc;
%pang_comment

root_path = 'D:\MatlabWorkspace\Get_hks\result_mat\';
label_path = 'SegmentationClass/';
label_result_path = 'img_label_result_20201105/';

if(exist(label_result_path,'dir')==0)
    mkdir(label_result_path);
    disp('makedir: label_result_path')
end

mat_list = dir(strcat(root_path,'*.mat'));
mat_list_length = length(mat_list);

label_list = dir(strcat(label_path,'*.png'));
label_list_length = length(label_list);


for i = 1:1:label_list_length
% for i = 570:1:label_list_length
%for i = 125:1:125
    mat_yuan_name = mat_list(i).name;
    mat_yuan_name = mat_yuan_name(1:11);
    label_image_name = label_list(i).name;
    label_image_name = label_image_name(1:11);
    if(strcmp(mat_yuan_name, label_image_name))
        mat_name = strcat(root_path,mat_list(i).name);
        %disp(mat_list(i).name);
        temp_name = mat_list(i).name;
        disp(strrep(temp_name,'.mat',''));
        
        load(mat_name);
        img_struct{1} = shape.result;
        
        label_name = strcat(label_path,label_list(i).name);
        %disp(label_list(i).name);
        
        label_img = imread(label_name);
        label = zeros(1024, 1);
        for j = 1:1:1024
            label(j) = label_img(img_struct{1}.X(j), img_struct{1}.Y(j));
        end
        label_struct{1}.X = img_struct{1}.X;
        label_struct{1}.Y = img_struct{1}.Y;
        label_struct{1}.label = label;
        
        label_mat_path = strcat(label_result_path,mat_list(i).name);
        %label_struct{1}.sihks = img_struct{1}.sihks;
        save(label_mat_path,'label_struct');
       
        clear img_struct;
        clear label_struct;
    else
        break;
    end
end