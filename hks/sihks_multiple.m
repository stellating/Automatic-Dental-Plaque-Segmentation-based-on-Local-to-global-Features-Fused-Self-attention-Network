close all;%关闭所有的结果窗口
clear all;%清空工作区缓存变量
clc;%清空命令行窗口缓存变量  


%%

root_path = 'result_mat/';%mat的路径
all_dst_path = 'hks_result_20201024/';%全部结果的路径

%如果文件夹不存在，则创建文件夹
if(exist(all_dst_path, 'dir')==0)
    mkdir(all_dst_path);
    disp('创建文件夹成功！');
end

%%
mat_list = dir(strcat(root_path, '*.mat'));
mat_list_length = length(mat_list);
%%
%for temp_i = 1:1:2
for temp_i = 1:1:mat_list_length
%for temp_i = 574:1:mat_list_length
    mat_name = strcat(root_path,mat_list(temp_i).name);
    %mat_name = 'imgmat/2019_000002.mat';
    disp('The mat name is: ');
    disp(mat_list(temp_i).name);
    load(mat_name); %把图片数据load进来，load进来的数据为img_struct
    img_struct{1} = shape.result;
    TRIV_3dims = delaunay(img_struct{1}.X, img_struct{1}.Y);
    img_struct{1}.TRIV = TRIV_3dims;
    
    K = 100;            % number of eigenfunctions
    alpha = 2;          % log scalespace basis
    
    T1 = [5:0.5:16];    % time scales for HKS
    T2 = [1:0.2:20];    % time scales for SI-HKS
    Omega = 2:20;       % frequencies for SI-HKS
    
    
    %%
    % compute cotan Laplacian  %计算余切拉普拉斯算子
    [img_struct{1}.W, img_struct{1}.A] = mshlp_matrix(img_struct{1});
    %从网格中计算 Laplace-Beltrami矩阵
    img_struct{1}.A = spdiags(img_struct{1}.A,0,size(img_struct{1}.A,1),size(img_struct{1}.A,1));
    %提取并创建稀疏带状和对角矩阵
    
    % compute eigenvectors/values
    [img_struct{1}.evecs,img_struct{1}.evals] = eigs(img_struct{1}.W,img_struct{1}.A,K,'SM');
    %计算特征值和特征向量  %返回K个最大特征值 %求广义矩阵的特征值
    img_struct{1}.evals = -diag(img_struct{1}.evals);
    %获取对角线元素
    
    % compute descriptors
    img_struct{1}.hks = myhks(img_struct{1}.evecs,img_struct{1}.evals,alpha.^T1);
    %计算hks特征
    [img_struct{1}.sihks, img_struct{1}.schks] = mysihks(img_struct{1}.evecs,img_struct{1}.evals,alpha,T2,Omega);
    %计算sihks特征
    %注意mysihks和sihks函数一模一样，只有函数名不同，sihks是内置函数，会冲突，所以选择mysihks解决冲突
    
    %%    
    disp('success!');
    disp(mat_list(temp_i).name(1:end-4));
    all_result_mat_path = strcat(all_dst_path,mat_list(temp_i).name);
    save(all_result_mat_path,'img_struct');
    
    %%
    clear img_struct;
    clear shape;
end
