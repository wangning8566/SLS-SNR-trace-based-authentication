% 交叉熵损失函数，此处的logits是未经过sigmoid激活的
% https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
function result = sigmoid_cross_entropy1(logits, labels)
    result = max(logits, 0) - logits .* labels + log(1 + exp(-abs(logits)));
    result = mean(result);
end