% discriminator��bp�������bp�漰���Ը�����������
% �����������ṹ��������ȣ����漰��bp�ĸ��ģ�����weights��biases�ĸ�������Ҫ����bp
% Ϊ�˸���w,b������Ҫ�����յ�loss��w��b��ƫ�������в��������w��bƫ�������м������̵Ľ��
function nn = nnbp_d(nn, y_h, y)
    % d��ʾ�в�в�������յ�loss�Ը���δ����ֵ��z����ƫ����ƫ�����ļ�����Ҫ������ʽ�󵼷���-�Լ��ֶ��Ƴ���
    n = nn.layers_count;
    % ���һ��Ĳв�
    nn.layers{n}.d = delta_sigmoid_cross_entropy1(y_h, y);
    for i = n-1:-1:2
        d = nn.layers{i+1}.d;
        w = nn.layers{i+1}.w;
        z = nn.layers{i}.z;
        % ÿһ��Ĳв��Ƕ�ÿһ���δ����ֵ��ƫ�����������Ǻ�һ��Ĳв����w,�ٳ��϶Լ���ֵ��δ����ֵ��ƫ����
        nn.layers{i}.d = d*w' .* delta_relu1(z);    
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