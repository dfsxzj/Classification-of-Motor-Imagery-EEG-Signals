function d=goujian(vec,wei)
%vec向量a*b，节点*数据
%wei等于1时，马氏距离，等于2时，欧氏距离，等于3时，相位滞后指数
a=size(vec,1);
b=size(vec,2);
d=zeros(a,a);
if wei==2
    for i=1:a-1
        subij=zeros(a-i,b);
        subij=vec(i,:)-vec(i+1:a,:);
        d(i,i+1:a)=arrayfun(@(k)norm(subij(k,:)),1:a-i);
    end
    d=d+d';
end
if wei==1
    Cov=cov(vec)+0.0001;  % 
    s=eye(size(Cov,1))/Cov;
    for i=1:a
        for j=i+1:a
            subij=zeros(1,b);
            subij=vec(i,:)-vec(j,:);
            d(i,j)=sqrt(abs(subij*s*subij'));
        end
    end
    d=d+d';
end
if wei==3
    vec=vec';
    ch=size(vec,2); % column should be channel
    %%%%%% Hilbert transform and computation of phases
    % for i=1:ch
    %     phi1(:,i)=angle(hilbert(X(:,i)));
    % end
    phi1=angle(hilbert(vec));
    PLI=ones(ch,ch);
    for ch1=1:ch-1
        for ch2=ch1+1:ch
            %%%%%% phase lage index
            PDiff=phi1(:,ch1)-phi1(:,ch2); % phase difference
            PLI(ch1,ch2)=abs(mean(sign(sin(PDiff)))); % only count the asymmetry
            PLI(ch2,ch1)=PLI(ch1,ch2);
        end
    end
    for i=1:ch
        PLI(i,i)=0;
    end
    d=PLI;
    end
end