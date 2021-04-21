close all;%关闭所有的结果窗口
clear all;%清空工作区缓存变量
clc;%清空命令行窗口缓存变量  

%%
root_path = 'xxx/';%mat的路径
all_dst_path = 'xxx/';%全部结果的路径

%如果文件夹不存在，则创建文件夹
if(exist(all_dst_path, 'dir')==0)
    mkdir(all_dst_path);
    disp('创建文件夹成功！');
end
%%
mat_list = dir(strcat(root_path, '*.mat'));
mat_list_length = length(mat_list);

for temp_i = 1:1:mat_list_length
    mat_name = strcat(root_path,mat_list(temp_i).name);
    disp('The mat name is: ');
    disp(mat_list(temp_i).name);
    load(mat_name); %把图片数据load进来，load进来的数据为img_struct
    
    field = {'A','W','TRIV','evecs','evals','sihks','schks'};
    img_struct{1} = rmfield(img_struct{1},field);
    
    disp('success!');
    disp(mat_list(temp_i).name(1:end-4));
    all_result_mat_path = strcat(all_dst_path,mat_list(temp_i).name);
    save(all_result_mat_path,'img_struct');
end