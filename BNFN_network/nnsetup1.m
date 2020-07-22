% 根据所给的结构建立网络
%build a network depend on the given 
function nn = nnsetup1(architecture)
    nn.architecture   = architecture;
    nn.layers_count = numel(nn.architecture);
    % t,beta1,beta2,epsilon,nn.layers{i}.w_m,nn.layers{i}.w_v,nn.layers{i}.b_m,nn.layers{i}.b_v是应用adam算法更新网络所需的变量
    nn.t = 0;
    nn.beta1 = 0.9;
    nn.beta2 = 0.999;
    nn.epsilon = 10^(-8);
    % 假设结构为[100, 512, 784]，则有3层，输入层100，两个隐藏层：100*512，512*784, 输出为最后一层的a值（激活值）
    %assum that there is struction [100, 512, 784] input layer is 100; two
    %hide layer : 100*512，512*784, the output is the last layer value
    %which is a (active value)
    for i = 2 : nn.layers_count   
        nn.layers{i}.w = normrnd(0, 0.02, nn.architecture(i-1), nn.architecture(i));
        nn.layers{i}.b = normrnd(0, 0.02, 1, nn.architecture(i));
        nn.layers{i}.w_m = 0;
        nn.layers{i}.w_v = 0;
        nn.layers{i}.b_m = 0;
        nn.layers{i}.b_v = 0;
    end
end