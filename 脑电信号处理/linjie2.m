function w= linjie2(d,k,e)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
n=size(d,2);
w=zeros(n,n);
d=squeeze(d);
%% 选大的连接
for i=1:n   
    d=d-eye(n);
    [a,b]=sort(d(i,:),'descend');
    if a(k)>=e
        w(i,:)=d(i,:)>e;
    elseif a(k)<e
        w(i,b(1:k))=1;
    end
end
w=ceil((w+w')/2);


