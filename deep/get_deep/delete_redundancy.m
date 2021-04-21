close all;%�ر����еĽ���
clear all;%��չ���������
clc;%��������д��ڻ������  

%%
root_path = 'deep_3_hks_lbp_as_z_result_20201104/';%mat��·��
all_dst_path = 'deep_3_hks_lbp_as_z_result_20201104_slim_high/';%ȫ������·��

%����ļ��в����ڣ��򴴽��ļ���
if(exist(all_dst_path, 'dir')==0)
    mkdir(all_dst_path);
    disp('�����ļ��гɹ���');
end
%%
mat_list = dir(strcat(root_path, '*.mat'));
mat_list_length = length(mat_list);

for temp_i = 1:1:mat_list_length
%for temp_i = 1:1:100
    mat_name = strcat(root_path,mat_list(temp_i).name);
    disp('The mat name is: ');
    disp(mat_list(temp_i).name);
    load(mat_name); %��ͼƬ���load������load���������Ϊimg_struct
    
    field = {'A','W','TRIV','evecs','evals','sihks','schks'};
    img_struct{1} = rmfield(img_struct{1},field);
    
    disp('success!');
    disp(mat_list(temp_i).name(1:end-4));
    all_result_mat_path = strcat(all_dst_path,mat_list(temp_i).name);
    save(all_result_mat_path,'img_struct','-v7.3');
end