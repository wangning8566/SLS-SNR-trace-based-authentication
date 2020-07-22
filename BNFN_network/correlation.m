function r= correlation( a,b )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
c=abs(sum((a-mean(a)).*(b-mean(b))));
d1=abs(sum((a-mean(a)).^2));
d2=abs(sum((b-mean(b)).^2));

d=sqrt(d1)*sqrt(d2);
r=c/d;

end







