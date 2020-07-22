function [sia_nn,E,error]=sia_training_loss(sia_nn, networkOutput,sampleOutput,numberOfSample)


%%%%%%%%%%%%%%% residual%%%%%%%%%%%%%%%%%%%
    probs=exp_score(networkOutput);   
    loss1 =loss_function1(sampleOutput,probs,numberOfSample);
    error = probs-sampleOutput;%the error between the real and network
    
  E=loss1
    sia_nn.errorHistory = [sia_nn.errorHistory E];

end