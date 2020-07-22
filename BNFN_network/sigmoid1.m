% sigmoid激活函数
function output = sigmoid1(x)
    output =1./(1+exp(-x));
end
% relu
function output = relu(x)
    output = max(x, 0);
end
% relu对x的导数
function output = delta_relu(x)
    output = max(x,0);
    output(output>0) = 1;
end
% 交叉熵损失函数，此处的logits是未经过sigmoid激活的
% https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
function result = sigmoid_cross_entropy(logits, labels)
    result = max(logits, 0) - logits .* labels + log(1 + exp(-abs(logits)));
    result = mean(result);
end
% sigmoid_cross_entropy对logits的导数，此处的logits是未经过sigmoid激活的
function result = delta_sigmoid_cross_entropy(logits, labels)
    temp1 = max(logits, 0);
    temp1(temp1>0) = 1;
    temp2 = logits;
    temp2(temp2>0) = -1;
    temp2(temp2<0) = 1;
    result = temp1 - labels + exp(-abs(logits))./(1+exp(-abs(logits))) .* temp2;
end
% 根据所给的结构建立网络
function nn = nnsetup(architecture)
    nn.architecture   = architecture;
    nn.layers_count = numel(nn.architecture);
    % t,beta1,beta2,epsilon,nn.layers{i}.w_m,nn.layers{i}.w_v,nn.layers{i}.b_m,nn.layers{i}.b_v是应用adam算法更新网络所需的变量
    nn.t = 0;
    nn.beta1 = 0.9;
    nn.beta2 = 0.999;
    nn.epsilon = 10^(-8);
    % 假设结构为[100, 512, 784]，则有3层，输入层100，两个隐藏层：100*512，512*784, 输出为最后一层的a值（激活值）
    for i = 2 : nn.layers_count   
        nn.layers{i}.w = normrnd(0, 0.02, nn.architecture(i-1), nn.architecture(i));
        nn.layers{i}.b = normrnd(0, 0.02, 1, nn.architecture(i));
        nn.layers{i}.w_m = 0;
        nn.layers{i}.w_v = 0;
        nn.layers{i}.b_m = 0;
        nn.layers{i}.b_v = 0;
    end
end
% 前向传递
function nn = nnff1(nn, x)
    nn.layers{1}.a = x;
    for i = 2 : nn.layers_count
        input = nn.layers{i-1}.a;
        w = nn.layers{i}.w;
        b = nn.layers{i}.b;
        nn.layers{i}.z = input*w + repmat(b, size(input, 1), 1);
        if i ~= nn.layers_count
            nn.layers{i}.a = relu(nn.layers{i}.z);
        else
            nn.layers{i}.a = sigmoid(nn.layers{i}.z);
        end
    end
end
% discriminator的bp，下面的bp涉及到对各个参数的求导
% 如果更改网络结构（激活函数等）则涉及到bp的更改，更改weights，biases的个数则不需要更改bp
% 为了更新w,b，就是要求最终的loss对w，b的偏导数，残差就是在求w，b偏导数的中间计算过程的结果
function nn = nnbp_d(nn, y_h, y)
    % d表示残差，残差就是最终的loss对各层未激活值（z）的偏导，偏导数的计算需要采用链式求导法则-自己手动推出来
    n = nn.layers_count;
    % 最后一层的残差
    nn.layers{n}.d = delta_sigmoid_cross_entropy(y_h, y);
    for i = n-1:-1:2
        d = nn.layers{i+1}.d;
        w = nn.layers{i+1}.w;
        z = nn.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        nn.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = nn.layers{i}.d;
        a = nn.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        nn.layers{i}.dw = a'*d / size(d, 1);
        nn.layers{i}.db = mean(d, 1);
    end
end
% generator的bp
function g_net = nnbp_g(g_net, d_net)
    n = g_net.layers_count;
    a = g_net.layers{n}.a;
    % generator的loss是由label_fake得到的，(images_fake过discriminator得到label_fake)
    % 对g进行bp的时候，可以将g和d看成是一个整体
    % g最后一层的残差等于d第2层的残差乘上(a .* (a_o))
    g_net.layers{n}.d = d_net.layers{2}.d * d_net.layers{2}.w' .* (a .* (1-a));
    for i = n-1:-1:2
        d = g_net.layers{i+1}.d;
        w = g_net.layers{i+1}.w;
        z = g_net.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        g_net.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = g_net.layers{i}.d;
        a = g_net.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        g_net.layers{i}.dw = a'*d / size(d, 1);
        g_net.layers{i}.db = mean(d, 1);
    end
end
% 应用梯度
% 使用adam算法更新变量，可以参考：
% https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
function nn = nnapplygrade(nn, learning_rate)
    n = nn.layers_count;
    nn.t = nn.t+1;
    beta1 = nn.beta1;
    beta2 = nn.beta2;
    lr = learning_rate * sqrt(1-nn.beta2^nn.t) / (1-nn.beta1^nn.t);
    for i = 2:n
        dw = nn.layers{i}.dw;
        db = nn.layers{i}.db;
        % 下面的6行代码是使用adam更新weights与biases
        nn.layers{i}.w_m = beta1 * nn.layers{i}.w_m + (1-beta1) * dw;
        nn.layers{i}.w_v = beta2 * nn.layers{i}.w_v + (1-beta2) * (dw.*dw);
        nn.layers{i}.w = nn.layers{i}.w - lr * nn.layers{i}.w_m ./ (sqrt(nn.layers{i}.w_v) + nn.epsilon);
        nn.layers{i}.b_m = beta1 * nn.layers{i}.b_m + (1-beta1) * db;
        nn.layers{i}.b_v = beta2 * nn.layers{i}.b_v + (1-beta2) * (db.*db);
        nn.layers{i}.b = nn.layers{i}.b - lr * nn.layers{i}.b_m ./ (sqrt(nn.layers{i}.b_v) + nn.epsilon); 
    end
end
% 保存图片，便于观察generator生成的images_fake
function save_images(images, count, path)
    n = size(images, 1);
    row = count(1);
    col = count(2);
    I = zeros(row*28, col*28);
    for i = 1:row
        for j = 1:col
            r_s = (i-1)*28+1;
            c_s = (j-1)*28+1;
            index = (i-1)*col + j;
            pic = reshape(images(index, :), 28, 28);
            I(r_s:r_s+27, c_s:c_s+27) = pic;
        end
    end
    imwrite(I, path);
end