function [finA]=nufuzz2(C,c0,ir)
A=C(C(:,end)==1,1:end-1);
B= C(C(:,end)~=1,1:end-1);

[x,y]=size(A);
[x1, y1]=size(B);

C=[A,ones(x,1);B,-ones(x1,1)]; % redefining C bcz if vectorization the order may change hence to keep it as per the order of A and B obtained
[no_input,no_col]=size(C);
obs = C(:,end);


s = sum(A,1)/x;
h = sum(B,1)/x1;

[Bcol,~]=size(B);
DiffB=B-repmat(h,[Bcol,1]);
distancec1=sqrt(diag(DiffB*DiffB'));

[val ind]=max(distancec1);
rn=val;


diff = s-h ;
db= sqrt(diff * diff');

for i = 1:no_input
    diff = C(i,1:no_col-1)-s ;
    dist1= sqrt(diff * diff');
    diff = C(i,1:no_col-1)-h ;
    dist2= sqrt(diff * diff');
    if(obs(i) == 1)
        memb(i,1)=1;
    else
        memb(i,1)=(1/(ir+1))+(ir/(ir+1))*(exp(c0*((dist1-dist2)/db-dist2/rn))-exp(-2*c0))/(exp(c0)-exp(-2*c0));
    end
end
finA=[C memb];
end
