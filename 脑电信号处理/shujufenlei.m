load('MIsuoyin.mat');

datas=zeros(168*15,64,700);
labels=zeros(168*15,1);
for i=1:14
    for j=1:12
        x(i,j,31)=size(data,4);
        for k=2:2:30
            temp1=squeeze(squeeze(x(i,j,k)));
            temp2=squeeze(squeeze(x(i,j,k+1)));
            datas(12*15*i-12*15+j*15-15+k/2,:,temp1-temp1+1:temp2-temp1+1)=data(i,j,:,temp1:temp2);
            labels(12*i*15-12*15+j*15-15+k/2)=mod(j-1,4)+1;
        end
    end
end
