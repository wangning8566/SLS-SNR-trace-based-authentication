function [sia_nn]=sia_training_nnbf(sia_nn,sampleInput1,error,numberOfSample)

   %%%%%%%%%%%%update the weights%%%%%%%%%%%%%%%%%%%%
    dis=sia_nn.dis;
    delta3 =error;
    
    dW2 = delta3* dis';
    dB2 = delta3 * ones(numberOfSample, 1);
    
    W2_1=ones(sia_nn.size(end),sia_nn.size(end-1));
    %delta1{sia_nn.n-2} = W2_1' * delta3.*hiddenOutput1{end}.*(1 - hiddenOutput1{end});
    
    for j=sia_nn.n-2:-1:1
         if j==sia_nn.n-2
    delta1{j} = W2_1' * delta3.*sia_nn.a1{end}.*(1 - sia_nn.a1{end});
         else
    delta1{j} = sia_nn.W{j+1}' * delta1{j+1}.*sia_nn.a1{j+1}.*(1 - sia_nn.a1{j+1});
         end
    
   dW1{j} = delta1{j}*sia_nn.a1{j}';
    dB1{j}= delta1{j}*ones(numberOfSample, 1);
         
    end

    for k=1:sia_nn.n-1
    if k==sia_nn.n-1
    sia_nn.W{k} = sia_nn.W{k} -sia_nn.learningRate *dW2;
    sia_nn.B{k} = sia_nn.B{k} -sia_nn.learningRate *dB2;
    else
   sia_nn.W{k} = sia_nn.W{k} - sia_nn.learningRate *dW1{k};
   sia_nn.B{k} = sia_nn.B{k} - sia_nn.learningRate *dB1{k};
    end
    end
   

end