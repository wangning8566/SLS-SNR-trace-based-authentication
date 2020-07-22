function [sia_nn,networkOutput]=sia_training_nnff(sia_nn,sampleInput1,sampleInput2)

%batch_num=200;
numberOfSample=length(sampleInput1(1,:));

%sample_ind=randperm(numberOfSample); % label index

%sampleInput_1=sampleInput1(:,sample_ind);
%sampleInput_2=sampleInput2(:,sample_ind);
%label_0=sampleOutput(:,sample_ind);
%batch_1=sampleInput_1(:,1:batch_num);
%batch_2=sampleInput_2(:,1:batch_num);
%label_batch=label_0(:,1:batch_num);
sia_nn.a1{1}=sampleInput1;
sia_nn.a2{1}=sampleInput2;



   %%%%%%%%%%%%%%%%hidden layer %%%%%%%%%%% %%%%
 for n=1:sia_nn.n-2
        if n==1
    sia_nn.a1{n+1} = logsig(sia_nn.W{n} * sia_nn.a1{n} + repmat(sia_nn.B{n}, 1, numberOfSample));%the output of hide layer
    sia_nn.a2{n+1} = logsig(sia_nn.W{n} * sia_nn.a2{n} + repmat(sia_nn.B{n}, 1, numberOfSample));%the output of hide layer
        else
    sia_nn.a1{n+1} = logsig(sia_nn.W{n} * sia_nn.a1{n} + repmat(sia_nn.B{n}, 1, numberOfSample));%the output of hide layer
    sia_nn.a2{n+1} = logsig(sia_nn.W{n} * sia_nn.a2{n}  + repmat(sia_nn.B{n}, 1, numberOfSample));%the output of hide layer
            
        end
    end

   %%%%%%%%%%%%% output layer%%%%%%%%%%%%%%%%%%
sia_nn.dis=dis_channel(sia_nn.a1{end},sia_nn.a2{end},numberOfSample);
networkOutput= sia_nn.W{sia_nn.n-1} * sia_nn.dis + repmat(sia_nn.B{sia_nn.n-1}, 1, numberOfSample); %the output of the output layer

end

