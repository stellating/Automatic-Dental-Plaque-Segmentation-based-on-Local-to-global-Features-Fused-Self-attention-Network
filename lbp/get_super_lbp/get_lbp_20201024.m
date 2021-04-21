close all;
clear all;
clc;
%pang_comment
%该代码用来计算LBP的mat


root_path = '../Get_hks/hks_result_20201024/';
lbp_path = './dst4/';
lbp_result_path = 'LBP20201024/lbp_result_20201024/';
all_result_lbp_path = 'LBP20201024/all_result_hks_lbp_20201024/';

if(exist(lbp_result_path,'dir')==0)
    mkdir(lbp_result_path);
    disp('makedir: lbp_result_path')
end

if(exist(all_result_lbp_path,'dir')==0)
    mkdir(all_result_lbp_path);
    disp('makedir: all_result_lbp_path')
end


mat_list = dir(strcat(root_path,'*.mat'));
mat_list_length = length(mat_list);

lbp_list = dir(strcat(lbp_path,'*.png'));
lbp_list_length = length(lbp_list);


for i = 1:1:lbp_list_length
% for i = 1220:1:lbp_list_length
    %for i = 1:1:1
    hks_mat_name = mat_list(i).name;
    hks_mat_name = hks_mat_name(1:11);
    lbp_image_name = lbp_list(i).name;
    lbp_image_name = lbp_image_name(1:11);
    if(strcmp(hks_mat_name, lbp_image_name))
        mat_name = strcat(root_path,mat_list(i).name);
        disp(mat_list(i).name);
        temp_name = mat_list(i).name;
        %disp(strrep(temp_name,'.mat',''));
        
        load(mat_name);%get the img_struct
        
        lbp_name = strcat(lbp_path,lbp_list(i).name);
        disp(lbp_list(i).name);
        
        lbp_img1 = imread(lbp_name);
        lbp_img = im2double(lbp_img1);%get the lbp image
        %lbp_img = imread(lbp_name);
        
        lbp = zeros(1024, 1);
        lbp_mean = zeros(1024, 1);
        for j = 1:1:1024
            lbp(j) = lbp_img(img_struct{1}.X(j), img_struct{1}.Y(j));
            temp_lbp = 0.0;
            Pixels = img_struct{1}.Point{1,j};%current block pixels
            %disp('-------------------------------');
            for pi = 1:1:length(Pixels)
                %disp(lbp_img(Pixels(pi,1) + 1,Pixels(pi,2) + 1));
                temp_lbp = temp_lbp + lbp_img(Pixels(pi,1) + 1,Pixels(pi,2) + 1);
                %             disp('temp_lbp is');
                %             disp(temp_lbp);
            end
            lbp_mean(j) = temp_lbp / length(Pixels);
            clear Pixels;
        end
        lbp_struct{1}.X = img_struct{1}.X;
        lbp_struct{1}.Y = img_struct{1}.Y;
        lbp_struct{1}.lbp = lbp;
        lbp_struct{1}.lbp_mean = lbp_mean;
        
        lbp_mat_path = strcat(lbp_result_path,mat_list(i).name);
        save(lbp_mat_path,'lbp_struct');
        
        img_struct{1}.lbp = lbp;
        img_struct{1}.lbp_mean = lbp_mean;
        
        all_result_lbp_mat_path = strcat(all_result_lbp_path,mat_list(i).name);
        save(all_result_lbp_mat_path,'img_struct');
        
        clear img_struct;
        clear lbp_struct;
    else
        break;
    end
end