
function probs=exp_score(networkOutput)

 exp_scores=exp(networkOutput);
    probs=exp_scores./repmat(sum(exp_scores), 2, 1);
    
end