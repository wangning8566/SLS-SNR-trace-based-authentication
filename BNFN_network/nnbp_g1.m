% generator��bp
function g_net = nnbp_g1(g_net, d_net)
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
        g_net.layers{i}.d = d*w' .* delta_relu1(z);    
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