clc;clear;close all
%load('H_v.mat')

D1=load('N3M2.mat','-ascii');
D2=load('N2M2.mat','-ascii');
l1=1000;
for l=1:l1
sampleInput_1(:,l)=mapminmax(D1(l,:));
sampleInput_2(:,l)=mapminmax(D2(l,:));
end

errorHistory = [];%the sum of the error

sia_nn=sia_setup([36 100 2]);
numberOfSample=200;


[sampleInput1,sampleInput2,sampleOutput]=data_samples(numberOfSample,sampleInput_1,sampleInput_2);

for i=1:1000;
   
[sia_nn,networkOutput]=sia_training_nnff(sia_nn,sampleInput1,sampleInput2);
[sia_nn,E(i),error]=sia_training_loss(sia_nn,networkOutput,sampleOutput,numberOfSample);

   if E(i) < sia_nn.error0
        break;
    end
[sia_nn]=sia_training_nnbf(sia_nn,sampleInput1,error,numberOfSample);


 
end

D3=load('N2M2.mat','-ascii');
D4=load('N4M2.mat','-ascii');
 [Test_accury]=sia_testing(sia_nn,D3,D4)


















