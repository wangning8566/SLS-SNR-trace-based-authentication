function [Test_accury]=sia_testing(sia_nn,D3,D4)

l1=3000;
for l=1:l1
sampleInput_1_test(:,l)=mapminmax(D3(l,:));
sampleInput_2_test(:,l)=mapminmax(D4(l,:));
end

numberOftest=3000;
[input_test1,input_test2,output_test]=data_samples(numberOftest,sampleInput_1_test,sampleInput_2_test);


[sia_nn,testNetworkOutput]=sia_training_nnff(sia_nn,input_test1,input_test2);


testprobs=exp_score(testNetworkOutput);   
Y=round(testprobs);
Test_accury=sum(find(output_test(1,:)==Y(1,:))~=0)/(numberOftest);


end