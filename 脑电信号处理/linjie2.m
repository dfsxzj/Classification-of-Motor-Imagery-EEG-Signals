function w= linjie2(d,k,e)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
n=size(d,2);
w=zeros(n,n);
d=squeeze(d);
%% ѡ�������
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


