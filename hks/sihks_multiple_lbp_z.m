close all;%�ر����еĽ������
clear all;%��չ������������
clc;%��������д��ڻ������  
%pang_comment
%%
root_path = 'result_mat/';%mat��·��
all_dst_path = 'result_hks_lbp_as_z_20201024/';%ȫ�������·��

lbp_root_path = 'D:/MatlabWorkspace/Get_lbp/LBP20201024/lbp_result_20201024/';%lbp��Ϊzֵ

%����ļ��в����ڣ��򴴽��ļ���
if(exist(all_dst_path, 'dir')==0)
    mkdir(all_dst_path);
    disp('�����ļ��гɹ���');
end

%%
mat_list = dir(strcat(root_path, '*.mat'));
mat_list_length = length(mat_list);

lbp_mat_list = dir(strcat(lbp_root_path, '*.mat'));
%%
for temp_i = 1:1:mat_list_length
% for temp_i = 401:1:mat_list_length %2019_000292.mat��2019_000401.mat
    mat_name = strcat(root_path,mat_list(temp_i).name);
    %mat_name = 'imgmat/2019_000002.mat';
    disp('The mat name is: ');
    disp(mat_list(temp_i).name);
    load(mat_name); %��ͼƬ����load������load����������Ϊimg_struct
    
    
    
    lbp_mat_name = strcat(lbp_root_path,lbp_mat_list(temp_i).name);
    %mat_name = 'imgmat/2019_000002.mat';
    disp('The mat name is: ');
    disp(lbp_mat_list(temp_i).name);
    load(lbp_mat_name); %��lbp����load������load����������Ϊlbp_struct
    
    img_struct{1} = shape.result;
    
    img_struct{1}.oldZ = img_struct{1}.Z;
    img_struct{1}.Z = lbp_struct{1}.lbp_mean; %lbp_mean��ΪZֵ
    %img_struct{1}.Z = lbp_struct{1}.lbp;
    
    
    TRIV_3dims = delaunay(img_struct{1}.X, img_struct{1}.Y);
    img_struct{1}.TRIV = TRIV_3dims;
    
    K = 100;            % number of eigenfunctions
    alpha = 2;          % log scalespace basis
    
    T1 = [5:0.5:16];    % time scales for HKS
    T2 = [1:0.2:20];    % time scales for SI-HKS
    Omega = 2:20;       % frequencies for SI-HKS
    
    
    %%
    % compute cotan Laplacian  %��������������˹����
    [img_struct{1}.W, img_struct{1}.A] = mshlp_matrix(img_struct{1});
    %�������м��� Laplace-Beltrami����
    img_struct{1}.A = spdiags(img_struct{1}.A,0,size(img_struct{1}.A,1),size(img_struct{1}.A,1));
    %��ȡ������ϡ���״�ͶԽǾ���
    
    % compute eigenvectors/values
    [img_struct{1}.evecs,img_struct{1}.evals] = eigs(img_struct{1}.W,img_struct{1}.A,K,'SM');
    %��������ֵ����������  %����K���������ֵ %�������������ֵ
    img_struct{1}.evals = -diag(img_struct{1}.evals);
    %��ȡ�Խ���Ԫ��
    
    % compute descriptors
    img_struct{1}.hks = myhks(img_struct{1}.evecs,img_struct{1}.evals,alpha.^T1);
    %����hks����
    [img_struct{1}.sihks, img_struct{1}.schks] = mysihks(img_struct{1}.evecs,img_struct{1}.evals,alpha,T2,Omega);
    %����sihks����
    %ע��mysihks��sihks����һģһ����ֻ�к�������ͬ��sihks�����ú��������ͻ������ѡ��mysihks�����ͻ
    
    %%    
    disp('success!');
    disp(mat_list(temp_i).name(1:end-4));
    all_result_mat_path = strcat(all_dst_path,mat_list(temp_i).name);
    save(all_result_mat_path,'img_struct');
    
    %%
%     clear img_struct;
%     clear shape;
end
