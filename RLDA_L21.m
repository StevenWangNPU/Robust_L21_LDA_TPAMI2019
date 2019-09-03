function W = RLDA_new(X, y, m)
% X: data, each column is a sample 
% y: label vector
% d: reduced dimensions
% Reference: Towards Robust Discriminative Projections Learning via Non-greedy L2,1-Norm MinMax
% Written by Zheng Wang

%load parameters:

c = length(unique(y));
[d,n] = size(X);
X = X - repmat(mean(X,2),1,n);
for i=1:c
    Xc{i} = X(:,y==i);
    nc(i) = size(Xc{i},2);
    dc{i} = ones(nc(i),1);
end;
W = orth(rand(d,m));
alpha = zeros(m,n);
for iter  = 1:30
% calculate A and mk
    A = zeros(d,d);
    B = zeros(d,m);
    ob1 = 0;
    for i=1:c
        Xi = Xc{i};
        ni = nc(i);
        D = diag(0.5./dc{i});
        dd = diag(D);
        mi = Xi*dd/sum(dd);  % updata mk
        Xmi = Xi-mi*ones(1,ni);  % calculate ||Xi-mk||
        A = A + Xmi*D*Xmi';  % calculate A
        Xm{i} = Xmi;
    end;  
        A = max(A,A');
% calculate lambda and updata pik 

    for i = 1:n
    Xx = X(:,i);
    WXi = W'*Xx;
    a = sqrt(sum(WXi.*WXi,1));
    if a ~= 0
        alpha(:,i) = WXi./a;
    else
        alpha(:,i) = zeros(length(WXi),1);
    end
    B = B + Xx * alpha(:,i)';
    ob1 = ob1 + a;
    end
    
    for i=1:c
        Xmi = Xm{i};
        WX = W'*Xmi;
        dc{i} = sqrt(sum(WX.*WX,1));
        ob(i) = sum(dc{i});
    end;
    
    lambda = sum(ob)/ob1;
    obj(iter) = sum(ob)/ob1;
    OBJ1(iter) = sum(ob);
    OBJ2(iter) = ob1;

% updata W

     W = GPI(A,(lambda/2).*B,1);
%    W = GPI(A, B, 1);
   
% updata weights   
    for i=1:c
        Xmi = Xm{i};
        WX = W'*Xmi;
        dc{i} = sqrt(sum(WX.*WX,1) + eps);
    end;
end
% figure;
% plot(obj,'r');
% figure();
% plot(OBJ1,'b');
% figure;
% plot(OBJ2,'g');