% Ç°Ïò´«µİ
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

function output = sigmoid(x)
    output =1./(1+exp(-x));
end
% relu
function output = relu(x)
    output = max(x, 0);
end
