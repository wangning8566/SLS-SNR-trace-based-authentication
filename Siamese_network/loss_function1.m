function loss=loss_function1(sampleOutput,probs,numberOfSample)

    loss1=0;
    for n=1:numberOfSample
    
  exp_scores=-sampleOutput(:,n)'*log(probs(:,n));
   loss1=exp_scores+loss1;
    
end

loss=loss1/numberOfSample;

end