function [sampleInput1,sampleInput2,sampleOutput]=data_samples(numberOfSample,sampleInput_1,sampleInput_2)


%numberOfSample = 200; 

num=numberOfSample/2;
sample_ind=randperm(numberOfSample); % label index

input_1_1=sampleInput_1(:,1:num);
input_1_2=sampleInput_1(:,num+1:num*2);
input_2_1=sampleInput_2(:,1:num);
input_2_2=sampleInput_2(:,num+1:num*2);

 sampleInput1_1=[input_1_1,input_1_2];
 sampleInput1=sampleInput1_1(:,sample_ind);
 %sampleInput1=sampleInput1_1;
 sampleInput2_1=[input_1_2,input_2_1];
 sampleInput2=sampleInput2_1(:,sample_ind);
 %sampleInput2=sampleInput2_1;
% label
label1 =[ones(1,num),zeros(1,num);zeros(1,num),ones(1,num)]; % label
label=label1(:,sample_ind);

sampleOutput=label;

end