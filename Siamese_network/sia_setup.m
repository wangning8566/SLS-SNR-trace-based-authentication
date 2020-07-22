function [sia_nn]=sia_setup(architecture)
outputDimension = 2;
sia_nn.size   = architecture;
sia_nn.n      = numel(sia_nn.size);

%sia_nn.trainnum=10;
sia_nn.error0 = 0.65*10^(-2);%target error 
sia_nn.error0 = 0.65*10^(-2);%target error 
sia_nn.errorHistory = [];%the sum of the error
sia_nn.learningRate = 0.01;%learning rate


 % initialization
 
 for i=1:sia_nn.n-1
     if i==sia_nn.n-1
%sia_nn.W{i} =  0.05*rand(outputDimension, sia_nn.size(end)) ;%initial output weights
sia_nn.W{i} =  (rand(outputDimension, sia_nn.size(end))-0.5)* 2 * 4 * sqrt(6 / (sia_nn.size(end) + outputDimension));

%sia_nn.B{i} =0.05*rand(outputDimension,1) ;%bias
sia_nn.B{i} =(rand(outputDimension, 1)-0.5)* 2 * 4 * sqrt(6 / (sia_nn.size(end) + outputDimension));
     else
%sia_nn.W{i} = 0.05*rand(sia_nn.size(i+1), sia_nn.size(i)) ;%initial input weights
sia_nn.W{i} =(rand(sia_nn.size(i+1), sia_nn.size(i))-0.5)* 2 * 4 * sqrt(6 / (sia_nn.size(i+1) + sia_nn.size(i)));
%sia_nn.B{i}= 0.05*rand(sia_nn.size(i+1),1) ;%bias
sia_nn.B{i}=(rand(sia_nn.size(i+1), 1)-0.5)* 2 * 4 * sqrt(6 / (sia_nn.size(i+1) + sia_nn.size(i)));
     end   
     

     
     
 end
end

