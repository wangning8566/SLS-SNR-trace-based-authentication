% relu��x�ĵ���
function output = delta_relu1(x)
    output = max(x,0);
    output(output>0) = 1;
end