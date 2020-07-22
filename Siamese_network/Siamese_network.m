clc;clear;close all
%load('H_v.mat')

D1=load('N1M2.mat','-ascii');
D2=load('N2M2.mat','-ascii');
l1=3000;
for l=1:l1
sampleInput_1(:,l)=mapminmax(D1(l,:));
sampleInput_2(:,l)=mapminmax(D2(l,:));
end

% parameters
inputDimension = 36;

numberOfSample = 200; 
numberOfHiddenNeure1 = 100;
numberOfHiddenNeure2=2;

outputDimension =2;

num=numberOfSample/2;
sample_ind=randperm(numberOfSample); % label index

input_1_1=sampleInput_1(:,1:num);
input_1_2=sampleInput_1(:,num+1:num*2);
input_2_1=sampleInput_2(:,1:num);
input_2_2=sampleInput_2(:,num+1:num*2);
%input = [sampleInput_1(:,1:100),sampleInput_2(:,1:100)];
 sampleInput1_1=[input_1_1,input_1_2];
 sampleInput1=sampleInput1_1(:,sample_ind);
 sampleInput2_1=[input_1_2,input_2_1];
 sampleInput2=sampleInput2_1(:,sample_ind);
% label
label1 =[ones(1,num),zeros(1,num);zeros(1,num),ones(1,num)]; % label
%sample_len=length(output(1,:));

label=label1(:,sample_ind);
sampleOutput=label;

learningRate = 0.01;%learning rate
error0 = 0.65*10^(-2);%target error 
 % initialization
W1 = 0.05*rand(numberOfHiddenNeure1, inputDimension) ;%initial input weights
B1= 0.05*rand(numberOfHiddenNeure1,1) ;%bias
W2 =  0.05*rand(outputDimension, numberOfHiddenNeure2) ;%initial output weights
B2 =0.05*rand(outputDimension,1) ;%bias
errorHistory = [];%the sum of the error

for i = 1:1000
    hiddenOutput = logsig(W1 * sampleInput1 + repmat(B1, 1, numberOfSample));%the output of hide layer
    hiddenOutput2 = logsig(W1 * sampleInput2 + repmat(B1, 1, numberOfSample));%the output of hide layer
     [dis]=dis_channel(hiddenOutput,hiddenOutput2,numberOfSample);
    networkOutput= W2 * dis + repmat(B2, 1, numberOfSample); %the output of the output layer
    % residual
    probs=exp_score(networkOutput);   
    loss1(i) =loss_function1(sampleOutput,probs,numberOfSample);
    error = probs-sampleOutput;%the error between the real and network
    
  E=loss1(i)
    errorHistory = [errorHistory E];
    %error1(i)=error;
    if E < error0
        break;
    end
    %update the weights
    delta3 =error;
    W2_1=ones(numberOfHiddenNeure2,numberOfHiddenNeure1);
    delta1 = W2_1' * delta3.*hiddenOutput.*(1 - hiddenOutput);
    
    dW2 = delta3* dis';
    dB2 = delta3 * ones(numberOfSample, 1);
    dW1 = delta1 * sampleInput1';
    dB1 = delta1 * ones(numberOfSample, 1);
    
    W2 = W2 - learningRate *dW2;
    B2 = B2 -learningRate *dB2;
    W1 = W1 - learningRate *dW1;
    B1 = B1 - learningRate *dB1;
end

%% testing
D3=load('N3M2.mat','-ascii');
D4=load('N4M2.mat','-ascii');

l1=3000;
for l=1:l1
sampleInput_1_test(:,l)=mapminmax(D3(l,:));
sampleInput_2_test(:,l)=mapminmax(D4(l,:));
end

numberOftest=1000;

testinput_1_1=sampleInput_1_test(:,1:numberOftest);
testinput_1_2=sampleInput_1_test(:,numberOftest+1:numberOftest*2);
testinput_2_1=sampleInput_2_test(:,1:numberOftest);
testinput_2_2=sampleInput_2_test(:,numberOftest+1:numberOftest*2);

output_test =[zeros(1,numberOftest),ones(1,numberOftest);ones(1,numberOftest),zeros(1,numberOftest)]; % label

input_test1 = [testinput_1_2,testinput_1_1];
input_test2 = [testinput_2_2,testinput_1_2];

testHiddenOutput1 = logsig(W1 * input_test1 + repmat(B1, 1, 2*numberOftest));%the output of hide layer
testHiddenOutput2 = logsig(W1 * input_test2 + repmat(B1, 1, 2*numberOftest));%the output of hide layer
    
[dis]=dis_channel(testHiddenOutput1,testHiddenOutput2,2*numberOftest);
testNetworkOutput= W2 * dis + repmat(B2, 1, 2*numberOftest); %the output of the output layer
testprobs=exp_score(testNetworkOutput);   
Y=round(testprobs);
Test_accury=sum(find(output_test(1,:)==Y(1,:))~=0)/(numberOftest*2)