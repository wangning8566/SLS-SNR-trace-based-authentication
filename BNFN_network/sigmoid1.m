% sigmoid�����
function output = sigmoid1(x)
    output =1./(1+exp(-x));
end
% relu
function output = relu(x)
    output = max(x, 0);
end
% relu��x�ĵ���
function output = delta_relu(x)
    output = max(x,0);
    output(output>0) = 1;
end
% ��������ʧ�������˴���logits��δ����sigmoid�����
% https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
function result = sigmoid_cross_entropy(logits, labels)
    result = max(logits, 0) - logits .* labels + log(1 + exp(-abs(logits)));
    result = mean(result);
end
% sigmoid_cross_entropy��logits�ĵ������˴���logits��δ����sigmoid�����
function result = delta_sigmoid_cross_entropy(logits, labels)
    temp1 = max(logits, 0);
    temp1(temp1>0) = 1;
    temp2 = logits;
    temp2(temp2>0) = -1;
    temp2(temp2<0) = 1;
    result = temp1 - labels + exp(-abs(logits))./(1+exp(-abs(logits))) .* temp2;
end
% ���������Ľṹ��������
function nn = nnsetup(architecture)
    nn.architecture   = architecture;
    nn.layers_count = numel(nn.architecture);
    % t,beta1,beta2,epsilon,nn.layers{i}.w_m,nn.layers{i}.w_v,nn.layers{i}.b_m,nn.layers{i}.b_v��Ӧ��adam�㷨������������ı���
    nn.t = 0;
    nn.beta1 = 0.9;
    nn.beta2 = 0.999;
    nn.epsilon = 10^(-8);
    % ����ṹΪ[100, 512, 784]������3�㣬�����100���������ز㣺100*512��512*784, ���Ϊ���һ���aֵ������ֵ��
    for i = 2 : nn.layers_count   
        nn.layers{i}.w = normrnd(0, 0.02, nn.architecture(i-1), nn.architecture(i));
        nn.layers{i}.b = normrnd(0, 0.02, 1, nn.architecture(i));
        nn.layers{i}.w_m = 0;
        nn.layers{i}.w_v = 0;
        nn.layers{i}.b_m = 0;
        nn.layers{i}.b_v = 0;
    end
end
% ǰ�򴫵�
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
% discriminator��bp�������bp�漰���Ը�����������
% �����������ṹ��������ȣ����漰��bp�ĸ��ģ�����weights��biases�ĸ�������Ҫ����bp
% Ϊ�˸���w,b������Ҫ�����յ�loss��w��b��ƫ�������в��������w��bƫ�������м������̵Ľ��
function nn = nnbp_d(nn, y_h, y)
    % d��ʾ�в�в�������յ�loss�Ը���δ����ֵ��z����ƫ����ƫ�����ļ�����Ҫ������ʽ�󵼷���-�Լ��ֶ��Ƴ���
    n = nn.layers_count;
    % ���һ��Ĳв�
    nn.layers{n}.d = delta_sigmoid_cross_entropy(y_h, y);
    for i = n-1:-1:2
        d = nn.layers{i+1}.d;
        w = nn.layers{i+1}.w;
        z = nn.layers{i}.z;
        % ÿһ��Ĳв��Ƕ�ÿһ���δ����ֵ��ƫ�����������Ǻ�һ��Ĳв����w,�ٳ��϶Լ���ֵ��δ����ֵ��ƫ����
        nn.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % �������Ĳв�֮�󣬾Ϳ��Ը��ݲв��������loss��weights��biases��ƫ����
    for i = 2:n
        d = nn.layers{i}.d;
        a = nn.layers{i-1}.a;
        % dw�Ƕ�ÿ���weights����ƫ���������
        nn.layers{i}.dw = a'*d / size(d, 1);
        nn.layers{i}.db = mean(d, 1);
    end
end
% generator��bp
function g_net = nnbp_g(g_net, d_net)
    n = g_net.layers_count;
    a = g_net.layers{n}.a;
    % generator��loss����label_fake�õ��ģ�(images_fake��discriminator�õ�label_fake)
    % ��g����bp��ʱ�򣬿��Խ�g��d������һ������
    % g���һ��Ĳв����d��2��Ĳв����(a .* (a_o))
    g_net.layers{n}.d = d_net.layers{2}.d * d_net.layers{2}.w' .* (a .* (1-a));
    for i = n-1:-1:2
        d = g_net.layers{i+1}.d;
        w = g_net.layers{i+1}.w;
        z = g_net.layers{i}.z;
        % ÿһ��Ĳв��Ƕ�ÿһ���δ����ֵ��ƫ�����������Ǻ�һ��Ĳв����w,�ٳ��϶Լ���ֵ��δ����ֵ��ƫ����
        g_net.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % �������Ĳв�֮�󣬾Ϳ��Ը��ݲв��������loss��weights��biases��ƫ����
    for i = 2:n
        d = g_net.layers{i}.d;
        a = g_net.layers{i-1}.a;
        % dw�Ƕ�ÿ���weights����ƫ���������
        g_net.layers{i}.dw = a'*d / size(d, 1);
        g_net.layers{i}.db = mean(d, 1);
    end
end
% Ӧ���ݶ�
% ʹ��adam�㷨���±��������Բο���
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
        % �����6�д�����ʹ��adam����weights��biases
        nn.layers{i}.w_m = beta1 * nn.layers{i}.w_m + (1-beta1) * dw;
        nn.layers{i}.w_v = beta2 * nn.layers{i}.w_v + (1-beta2) * (dw.*dw);
        nn.layers{i}.w = nn.layers{i}.w - lr * nn.layers{i}.w_m ./ (sqrt(nn.layers{i}.w_v) + nn.epsilon);
        nn.layers{i}.b_m = beta1 * nn.layers{i}.b_m + (1-beta1) * db;
        nn.layers{i}.b_v = beta2 * nn.layers{i}.b_v + (1-beta2) * (db.*db);
        nn.layers{i}.b = nn.layers{i}.b - lr * nn.layers{i}.b_m ./ (sqrt(nn.layers{i}.b_v) + nn.epsilon); 
    end
end
% ����ͼƬ�����ڹ۲�generator���ɵ�images_fake
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