function [dis]=dis_channel(hiddenOutput1,hiddenOutput2,numberOfSample)

for n=1:numberOfSample

%B2(n)= pdist2(hiddenOutput1(:,n)',hiddenOutput2(:,n)','cosine');

A1(n)= pdist2(hiddenOutput1(:,n)',hiddenOutput2(:,n)','cosine')*100;


A2(n)= pdist2(hiddenOutput1(:,1)',hiddenOutput2(:,n)','correlation')*10;




end
dis=[A1;A2];
end