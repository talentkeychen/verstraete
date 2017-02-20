function timeevolution_verstrate(N,D,precision)
%N=10;
%D=5;
%precision=10^(-5);
dt=0.03;
jflipped=5;

%magnetization
oset=cell(1,N);
sx=[0,1;1,0];sy=[0,-1i;1i,0];sz=[1,0;0,-1];id=eye(2);
for j=1:N
    oset{1,j}=id;
end
oset{1,jflipped}=sz;

%time evolution operator
h=kron(sx,sx)+kron(sy,sy)+kron(sz,sz);
w=expm(-1i*dt*h);
w=reshape(w,[2,2,2,2]);
w=permute(w,[1,3,2,4]);
w=reshape(w,[4,4]);
[U,S,V]=svd2(w);
eta=size(S,1);
U=U*sqrt(S);
V=sqrt(S)*V;
U=reshape(U,[2,2,eta]);
U=permute(U,[4,3,2,1]);
V=reshape(V,[eta,2,2]);
V=permute(V,[1,4,3,2]);
I=reshape(id,[1,1,2,2]);
mpo_even=cell(1,N);
mpo_odd=cell(1,N);

for j=1:N
    mpo_even{j}=I;
    mpo_odd{j}=I;
end

for j=1:2:(N-1)
    mpo_odd{j}=U;
    mpo_odd{j+1}=V;
end

for j=2:2:(N-1)
    mpo_even{j}=U;
    mpo_even{j+1}=V;
end

%starting state (one spin flipped)
mps0=cell(1,N);
for j=1:N
    if j==jflipped
        state=[0;1];
    else
        state=[1;0];
    end
    mps0{j}=reshape(state,[1,1,2]);
end

%time evolution
mps=mps0;
mzvalues=[];

for step=1:50
    fprintf('Step %2d: \n',step);
    mps=reduceD(mps,mpo_even,D,precision);
    mps=reduceD(mps,mpo_odd,D,precision);
    mz=expectationvalue(mps,oset);
    mzvalues=[mzvalues,mz];
    fprintf('mz=%g\n',mz);
end
end

   
function mpsB=reduceD(mpsA,mpoX,DB,precision)

N=length(mpsA);
d=size(mpsA{1},3);
mpsB=createrandommps(N,DB,d);
mpsB=prepare(mpsB);
%initialization of the storage
Cstorage=initCstorage(mpsB,mpoX,mpsA,N);
%optimization sweeps
while 1
    Kvalues=[];
    
    %cycle for 1 to N-1
    for j=1:(N-1)
        %optimization
        Cleft=Cstorage{j};
        Cright=Cstorage{j+1};
        A=mpsA{j};
        X=mpoX{j};
        [B,K]=reduceD2_onesite(A,X,Cleft,Cright);
        [B,U]=prepare_onesite(B,'lr');
        mpsB{j}=B;
        Kvalues=[Kvalues,K];
        
        % storage-update
        Cstorage{j+1}=updateCleft(Cleft,B,X,A);
    end
    
    %cycle for N to 2
    for j=N:(-1):2
        %optimization
        Cleft=Cstorage{j};
        Cright=Cstorage{j+1};
        A=mpsA{j};
        X=mpoX{j};
        [B,K]=reduceD2_onesite(A,X,Cleft,Cright);
        [B,U]=prepare_onesite(B,'rl');
        mpsB{j}=B;
        Kvalues=[Kvalues,K];
        
        %storage-update
        Cstorage{j}=updateCright(Cright,B,X,A);
    end
    
    if std(Kvalues)/abs(mean(Kvalues))<precision
        mpsB{1}=contracttensors(mpsB{1},3,2,U,2,1);
        mpsB{1}=permute(mpsB{1},[1,3,2]);
        break;
    end
end
end

function [Cstorage]=initCstorage(mpsB,mpoX,mpsA,N)

Cstorage=cell(1,N+1);
Cstorage{1}=1;
Cstorage{N+1}=1;
for k=N:-1:2
    if isempty(mpoX)
        X=[];
    else
        X=mpoX{k};
    end
    Cstorage{k}=updateCright(Cstorage{k+1},mpsB{k},X,mpsA{k});
end
end


function [Cleft]=updateCleft(Cleft,B,X,A)
d=size(B,3);
if isempty(X)
    X=reshape(eye(d),[1,1,d,d]);
end
Cleft=contracttensors(A,3,1,Cleft,3,3);
Cleft=contracttensors(X,4,[1,3],Cleft,4,[4,2]);
Cleft=contracttensors(conj(B),3,[1,3],Cleft,4,[4,2]);
end


%convention mpsA 1    2,                                3
%             up    3       mpsB    3            mpo 1      2
%                         down    1    2                 4
%this function is to update from the right to the left
function [Cright]=updateCright(Cright,B,X,A)
%A is on the top , B is on the bottom
d=size(B,3);
if isempty(X)
    X=reshape(eye(d),[1,1,d,d]);
end
Cright=contracttensors(A,3,2,Cright,3,3);
Cright=contracttensors(X,4,[3,2],Cright,4,[2,4]);
Cright=contracttensors(conj(B),3,[2,3],Cright,4,[4,2]);
end


function [mps]=prepare(mps)
N=length(mps);

for i=N:-1:2
    [mps{i},U]=prepare_onesite(mps{i},'rl'); %right canonial
    mps{i-1}=contracttensors(mps{i-1},3,2,U,2,1);
    mps{i-1}=permute(mps{i-1},[1,3,2]);
end
end

function [B,U,DB]=prepare_onesite(A,direction)

[D1,D2,d]=size(A);
switch direction
case 'lr'
    A=permute(A,[3,1,2]);
    A=reshape(A,[d*D1,D2]);
    [B,S,U]=svd2(A);
    DB=size(S,1);
    B=reshape(B,[d,D1,DB]);
    B=permute(B,[2,3,1]);
    U=S*U;
case 'rl'
    A=permute(A,[1,3,2]);
    A=reshape(A,[D1,d*D2]);
    [U,S,B]=svd2(A);
    DB=size(S,1);
    B=reshape(B,[DB,d,D2]);
    B=permute(B,[1,3,2]);
    U=U*S;
end
end

function [B,K]=reduceD2_onesite(A,X,Cleft,Cright)

Cleft=contracttensors(Cleft,3,3,A,3,1);
Cleft=contracttensors(Cleft,4,[2,4],X,4,[1,3]);

B=contracttensors(Cleft,4,[3,2],Cright,3,[2,3]);
B=permute(B,[1,3,2]);

b=reshape(B,[numel(B),1]);
K=-b'*b;
end


function [e,n]=expectationvalue(mps,hset)

[M,N]=size(hset);
d=size(mps{1},3);

%expectation value
e=0;
for m=1:M
    em=1;
    for j=N:-1:1
        h=hset{m,j};
        h=reshape(h,[1,1,d,d]);
        em=updateCright(em,mps{j},h,mps{j});
    end
    e=e+em;
end
%norm
n=1;
X=eye(d);
X=reshape(X,[1,1,d,d]);
for j=N:-1:1
    n=updateCright(n,mps{j},X,mps{j});
end

e=e/n;
end


%contract of part of indexs of tenters 
% works as 5 2 2 and 3 5 2 to 2 3
function [X,numindX]=contracttensors(X,numindX,indX,Y,numindY,indY)

Xsize=ones(1,numindX);
Xsize(1:length(size(X)))=size(X);
Ysize=ones(1,numindY);
Ysize(1:length(size(Y)))=size(Y);

indXl=1:numindX; indXl(indX)=[];
indYr=1:numindY; indYr(indY)=[];

sizeX1=Xsize(indXl);
sizeX=Xsize(indX);
sizeYr=Ysize(indYr);
sizeY=Ysize(indY);

if prod(sizeX)~=prod(sizeY)
    error('indX and indY are not of the same dimension.');
end

if isempty(indYr)
    if isempty(indXl)
        X=permute(X,[indX]);
        X=reshape(X,[1,prod(sizeX)]);
        
        Y=permute(Y,[indY]);
        Y=reshape(Y,[prod(sizeY),1]);
        
        X=X*Y;
        Xsize=1;
    
    else
        X=permute(X,[indXl,indX]);
        X=reshape(X,[prod(sizeX1),prod(sizeX)]);
        
        Y=permute(y,[indY]);
        Y=reshape(Y,[prod(sizeY),1]);
        
        X=X*Y;
        Xsize=Xsize(indXl);
        
        X=reshape(X,[Xsize,1]);
    end
else

X=permute(X,[indXl,indX]);
X=reshape(X,[prod(sizeX1),prod(sizeX)]);

Y=permute(Y,[indY,indYr]);
Y=reshape(Y,[prod(sizeY),prod(sizeYr)]);

X=X*Y;
Xsize=[Xsize(indXl),Ysize(indYr)];
end


numindX=length(Xsize);
X=reshape(X,[Xsize,1]);
end

function [U,S,V]=svd2(T)

[m,n]=size(T);
if m>=n
    [U,S,V]=svd(T,0);
else
    [V,S,U]=svd(T',0);
end
V=V';
end

function [mps]=createrandommps(N,D,d)
%N>2
mps=cell(1,N);
mps{1}=randn(1,D,d)/sqrt(D);
mps{N}=randn(D,1,d)/sqrt(D);
for k=2:N-1
    mps{k}=randn(D,D,d)/sqrt(D);
end
end
