%--------------------
% Distant Supervision for Relation Extraction with Matrix Completion
%--------------------
%-----------------------------
clear
clc
% load dataMatrix from NAACL 2013 dataset
Z0 = load('dataMatrix_NYT-13_7313_3278_1947_51');

const = getConst();

% feature and label matrices construction
Xtrain = Z0(1:const.ntrain, 1:const.fd);
Ytrain = Z0(1:const.ntrain, const.fd + 1 : const.fd + const.ld);
Xtest = Z0(const.ntrain + 1:const.ntest + const.ntrain, 1:const.fd);
Ytest = Z0(const.ntrain + 1:const.ntest + const.ntrain, const.fd + 1 : const.fd + const.ld);
% DRMC-1
Out_1 = DRMC_1(Xtrain, Ytrain, Xtest, Ytest);
% DRMC-b
Out_b = DRMC_b(Xtrain, Ytrain, Xtest, Ytest);
