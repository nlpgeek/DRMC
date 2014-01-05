function result = evaluation(Z, Ytest, bias)
    const = getConst();
    [demZR, demZC] = size(Z);
    [demYR, demYC] = size(Ytest);
    
    YPredict = Z(demZR - demYR + 1 : demZR, const.fd + 1 + bias:const.fd + const.ld + bias);
    [E, I] = sort(YPredict(:), 'descend');
    [m, n] = size(Ytest);
    preList = [];
    recList = [];
    prebase = m * n;
    recbase = sum(Ytest(:) > 0);
    for i = 10 : 5 : prebase
%         YPredict(I(1:i))
        match = sum(Ytest(I(1:i)) > 0);
        
        precision = match * 1.0 / i;
        recall = match * 1.0 / recbase;
        preList = [preList, precision];
        recList = [recList, recall];
    end
    
    result.preList = preList; 
    result.recList = recList;

end