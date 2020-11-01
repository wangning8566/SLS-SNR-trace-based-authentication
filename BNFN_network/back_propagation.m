clc;clear;close all

load('m3.mat');
load('m4.mat');
load('m5.mat');
device03=m3;
device02=m4;
device01=m5;
load('gan_data360.mat');
gan_data360 = images_fake;
%% train data
train_num=500;

[device11] = mapminmax(device01(1:train_num,:));
 %[device12] = mapminmax(device02(1:train_num,:));
 %[device13] = mapminmax(device03(1:train_num,:));
gan01 = mapminmax(gan_data360(1:train_num,:));
y=[ones(train_num,1);2*ones(train_num,1)];
%data_input = [device11;device12;device13]';
data_input = [device11]';
data_target = y';
data_input_n = [gan01]';
start_time_train1=cputime;
%% weight value
% these parameters can be obtained by nntool (matlab) when the device samples are used as
% the input samples and the corresponding labels are used as the output.

w1=[-0.54746 0.43397 -0.53106 -0.43192 0.13443 -0.19974 -0.32011 0.44142 -0.02169 -0.14121 0.25982 0.15837 0.45978 -0.23028 0.44508 0.19424 -0.53581 0.39441 0.31935;
 -0.51719 0.41164 0.049317 -0.32831 0.35667 0.20992 0.29504 0.52795 0.068772 0.12595 -0.15857 -0.58146 -0.55146 0.14377 -0.10982 -0.61651 0.19999 0.36182 -0.23481;
 -0.39063 -0.44927 -0.033227 -0.082233 0.83498 1.6291 -1.1807 -0.64504 0.32341 0.65853 0.010037 -0.53642 -1.1904 -0.36372 0.53143 0.12588 1.2171 -0.23305 -0.0012447;
 0.074442 0.57341 0.13453 0.5367 -0.34583 -0.28646 -0.075267 0.67945 0.44736 -0.27326 0.3366 0.17445 1.0558 -0.45643 0.080421 -0.72179 -0.097223 0.46711 -0.23813;
 -0.45768 0.50812 0.41925 0.28483 0.13379 0.35661 -0.47983 -0.062433 -0.10278 -0.27997 0.3895 0.041307 -0.29805 0.48603 0.27743 0.71067 0.53292 0.44307 0.22977;
 -1.2875 -0.81919 0.80939 -0.10978 1.4977 1.1444 -1.5585 -0.76659 0.11204 -0.40908 -0.37034 0.090987 -1.2767 0.1454 -0.092115 0.87174 1.8027 -0.78812 0.92774;
 0.02378 0.36908 0.099868 0.33912 -0.088042 -0.22777 0.52066 -0.015867 -0.63842 0.47559 0.13274 0.47364 0.31726 0.39672 -0.25538 0.14672 -0.95871 0.16184 0.21411;
 -0.083706 -0.14791 0.15411 -0.49829 0.54591 -0.24517 0.46377 -0.035767 -0.15408 -0.11488 0.35472 0.3434 0.51985 -0.30083 -0.3058 0.5692 -0.080133 0.58754 -0.36581;
 -0.21605 0.23244 -0.30175 0.54195 -0.20608 0.19406 -0.10589 -0.068116 -0.0072615 0.17553 -0.44261 0.36221 0.55484 0.37568 -0.62167 -0.58698 -0.30983 0.25858 0.35281;
 -0.020702 0.47401 0.67039 -0.36405 0.044846 0.19112 0.22287 0.43698 -0.24587 0.22629 0.29584 -0.26112 -0.58603 0.53855 0.020598 0.63143 0.15421 0.046885 0.51601];
w2=[-0.19309 -0.22596 2.6421 -0.93098 0.89433 3.8388 -0.76481 0.072374 -0.15026 0.84142];
b1=[1.5973;1.3039;-0.56674;0.26094;0.023322;-0.57629;-0.31299;-0.8861;-1.1856;1.5719];
b2=[-0.16072];

%% train function
q1=w1*data_input;
tempH1=q1+repmat(b1,1,train_num);
H1=tansig(tempH1);
tempH2=w2*H1+b2;
train_oneclass1=tempH2(1:train_num);
%train_oneclass1=tempH1(:,1:train_num);
 train_oneclass2=hypersphere_sample02(train_oneclass1);
tempH1_n=w1*data_input_n+repmat(b1,1,train_num);
H1_n=tansig(tempH1_n);
tempH2_n=w2*H1_n+b2;

 train_oneclass3 = tempH2_n;
 H3=[train_oneclass1,train_oneclass2(1:train_num/2), train_oneclass3(1:train_num/2)];
%% train oneclass
NumberofOutputNeurons=10;
NumberofTrainingData = train_num*2;
T=data_target(1:train_num*2);
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
%% calculate the output weight
Hidden_num = 5; % ELM input neuron number
ELM_in = 1; % feature dimension 
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(Hidden_num,ELM_in)*2-1;
BiasofHiddenNeurons=rand(Hidden_num,1);
tempH3=InputWeight*H3;
                                        %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH4=tempH3+BiasMatrix;

% active function
 H = 1 ./ (1 + exp(-tempH4));

% inverse function
OutputWeight=pinv(H') * T';

% train output
Y=(H' * OutputWeight)';     
save('one_class_model','InputWeight','BiasofHiddenNeurons','OutputWeight','w1','w2','b1','b2');