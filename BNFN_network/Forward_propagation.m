clc
clear
close all
%% data
device01=load('N1M2.mat','-ascii');
device02=load('N2M2.mat','-ascii');
device03=load('N3M2.mat','-ascii');
device1=device01(1:3000,:);
device2=device02(1:3000,:);
device3=device03(1:3000,:);
load('gan_data.mat');
%% 
%% test data
train_num=1000;
test_num=2000;
[device21] = mapminmax(device01(train_num+1:train_num+test_num,:));
[device22] = mapminmax(device02(train_num+1:train_num+test_num/2,:));
[device23] = mapminmax(gan_data(train_num+1:train_num+test_num/2,:));

y1=[ones(test_num,1);2*ones(test_num,1)];
data_t = [device21;device22;device23]';
data_target1 = y1';

%% weight value

load('one_class_model.mat');
%% feature extraction based on neural network
tempH1=w1*data_t+b1;
H1=tansig(tempH1);
tempH2=w2*H1+b2;

%% train oneclass
NumberofOutputNeurons=10;
NumberofTrainingData = test_num*2;
T=data_target1(1:test_num*2);
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(T,2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:NumberofTrainingData
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;
%% testing
 tempH3=InputWeight*tempH2;
 ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH4=tempH3+BiasMatrix;
  H = 1 ./ (1 + exp(-tempH4));
 Y=(H' * OutputWeight)';     
 
 %% test accuracy
 
    MissClassificationRate_Training=0;
    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        output(i)=label(label_index_actual);
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/NumberofTrainingData
