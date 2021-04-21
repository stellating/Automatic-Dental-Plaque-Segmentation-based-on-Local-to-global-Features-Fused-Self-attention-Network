close all;
clear all;
clc;
%pang_comment
%deep feature channel==3


%root_path = './all_result_hks_lbp_as_z_20200822/';
%deep_path = './feature700_512_121/';
%deep_result_path = './deep_512_result_20200822/';
%all_result_deep_path = './deep_512_hks_lbp_as_z_result_20200822/';

root_path = '/home/lenovo/pzn/MatlabWorkspace/datasets/all_lbp_as_z_hks/';
% deep_path = '/home/lenovo/pzn/seg/git_ocnet/runs/datasets1104/cnn512_train/';
deep_path = '/media/lenovo/pzn_Files of Study/cnn512_train/';
% deep_path = '/media/lenovo/pzn_Download Setup/cnn512_train/';
deep_result_path = 'deep_512_result_20201104/';
all_result_deep_path = 'deep_512_hks_lbp_as_z_result_20201104/';



if(exist(deep_result_path,'dir')==0)
    mkdir(deep_result_path);
    disp('makedir: deep_result_path')
end

if(exist(all_result_deep_path,'dir')==0)
    mkdir(all_result_deep_path);
    disp('makedir: all_result_deep_path')
end


mat_list = dir(strcat(root_path,'*.mat'));
mat_list_length = length(mat_list);
disp(mat_list_length);

deep_list = dir(strcat(deep_path,'*.mat'));
deep_list_length = length(deep_list);
disp(deep_list_length);

%%temp_j = 225; %2018_010107 %2019_000129 %2019_000682
temp_j = 1;
for i = 2732:1:mat_list_length
%%for i = 556:1:mat_list_length
% for i = 2102:1:mat_list_length
%for i = 2100:1:mat_list_length
%for i = 1:1:1
    mat_name_name = mat_list(i).name;
    mat_name_name = mat_name_name(1:end-4);
    
    deep_name_name = deep_list(temp_j).name;
    deep_name_name = deep_name_name(1:end-4);
    
    if(strcmp(mat_name_name,deep_name_name))
        mat_name = strcat(root_path,mat_list(i).name);
        disp(mat_list(i).name);
        temp_name = mat_list(i).name;
        %disp(strrep(temp_name,'.mat',''));
        
        load(mat_name);%get the img_struct
        
        deep_name = strcat(deep_path,deep_list(temp_j).name);
        disp(deep_list(temp_j).name);
        
        load(deep_name);%get the deep feature
        deep_feature_all = Bimage;
        
        %deep_feature = zeros(1024, 256);
        %deep_feature = zeros(1024, 3);
        deep_feature = zeros(1024, 512);
        
        for j = 1:1:1024
            deep_feature(j,:) = deep_feature_all(img_struct{1}.X(j), img_struct{1}.Y(j),:);
        end
        deepf_struct{1}.X = img_struct{1}.X;
        deepf_struct{1}.Y = img_struct{1}.Y;
        deepf_struct{1}.deep_feature = deep_feature;
        
        
        deep_mat_path = strcat(deep_result_path,mat_list(i).name);
        save(deep_mat_path,'deepf_struct');
        
        img_struct{1}.deep_feature = deep_feature;
        
        field = {'A','W','TRIV','evecs','evals','sihks','schks'};
        img_struct{1} = rmfield(img_struct{1},field);
        
        all_result_deep_mat_path = strcat(all_result_deep_path,mat_list(i).name);
        save(all_result_deep_mat_path,'img_struct');
        
                      
        
        clear img_struct;
        clear deepf_struct;
        temp_j = temp_j + 1;
    else
        break;
    end
end