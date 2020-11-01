clc;clear;close all

load('m4.mat');
load('m5.mat');

device02=m4(1:600,:);
device01=m5(1:1200,:);


%% test data
test_num=200;

[device21] = mapminmax(device01(1:test_num,:));
[device22] = mapminmax(device02(1:test_num,:));

y1=[ones(test_num,1);2*ones(test_num,1)];
data_t = [device21;device22]';
data_target1 = y1';

%% weight value

load('one_class_model.mat');
%% feature extraction based on neural network
q1=w1*data_t;
tempH1=q1+repmat(b1,1,test_num*2);
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
%% ELM testing
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