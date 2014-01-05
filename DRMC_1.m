function Out = DRMC_1(Xtrain, Ytrain, Xtest, Ytest)
    
    Out.r = [];
    Out.prec = [];
    Out.rec = [];
    
    [numOfXtrain, demXtrain] = size(Xtrain);
    [numOfYtrain, demYtrain] = size(Ytrain);
    
    X = [Xtrain; Xtest];
    [x, y] = find(X > 0);
    OmigaX = [x, y];
    [r, c] = size(OmigaX);
    numOfOmigaX = r;

    Y = [Ytrain; zeros(size(Ytest))];
    [x, y] = find(Y > 0);
    OmigaY = [x, y];
    [r, c] = size(OmigaY);
    numOfOmigaY = r;
    
    Z = [X, Y];
    [r, c] = size(Z);
    
    Z = [ones(r, 1), Z];
    
    params = getParams(Z, numOfOmigaY, numOfOmigaX);
    
    mu = params.mus;
    muf = params.muf;
    
    for i = 1 : params.maxOuterItr
        
        for j = 1 : params.maxInnerItr
            Zp = Z;
            gz = getGZ(params.lamda, numOfOmigaY, numOfOmigaX, OmigaY, OmigaX, Y, X, Z);
            A = Z - params.tauz * gz;

            [U, S, V] = svd(A);

            S = max(0,S-params.tauz * mu);
            Z = U * S * V';
            % projection to vector-1
            Z(:, 1) = ones(r, 1);
                       
            ra = rank(Z);

            Out.r = [Out.r; ra];
            result = evaluation(Z, Ytest, 1);
            Out.prec = [Out.prec; result.preList];
            Out.rec = [Out.rec; result.recList];
            
            if(ra <= params.rank_1)
                Out.Z = Z;
                return;
            end
            
            if (norm(Zp-Z, 'fro') / max(1.0, norm(Zp, 'fro')) <= params.tol)
                if(mu == muf)
                    Out.Z = Z;
                    return;
                else
                    break;
                end
            end
        end
        mu = max(mu * params.eta, muf);
    end    
end

    %
    % inner gradient function for Z(MC_1)
    %

function gz = getGZ(lamda, numOfOmigaY, numOfOmigaX, OmigaY, OmigaX, Y, X, Z)

    [rowZ, columnZ] = size(Z);
    [rowX, columnX] = size(X);

    gz = zeros(rowZ, columnZ);

    for k = 1 : numOfOmigaY
        i = OmigaY(k, 1);
        j = OmigaY(k, 2);
        gz(i, j + columnX + 1) = lamda / numOfOmigaY * (-Y(i, j) / (1 + exp(Y(i, j) * (Z(i, j + columnX + 1)))));        
    end

    for k = 1 : numOfOmigaX      
        i = OmigaX(k, 1);
        j = OmigaX(k, 2);
        gz(i, j + 1) = 1.0 / numOfOmigaX * (- X(i, j) / (1 + exp(X(i, j) * Z(i, j + 1))));
    end
end



