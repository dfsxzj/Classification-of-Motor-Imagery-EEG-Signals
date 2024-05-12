clear
%% Êý¾ÝÔ¤´¦Àí
load('MIprocesseddata.mat');
xindao = size(datas,2);
data=datas(:,:,1:640);
lab=labels;

%% Ð¡²¨±ä»»
num=size(data,1);
for c=1:20
    eval(strcat('temp',num2str(c),'=zeros(num,xindao);'))
end
for a=1:num
    for i=1:xindao
       [coe,lis]=eval(strcat('wavedec(squeeze(data(a,i,:))'',5,''db4'')'));
       coef_v1 = appcoef(coe,lis,'db4',5);%¦Ä²¨ 
       coef_v2 = detcoef(coe,lis,5);%¦È²¨
%         coef_v2 = lvbo(data(a,i,:),3.5,4,7,8,3,1,9);
       coef_v3 = detcoef(coe,lis,4);%¦Á²¨
%         coef_v3 = lvbo(data(a,i,:),7,8,13,14,6,1,15);
       coef_v4 = detcoef(coe,lis,3);%¦Â²¨
%         coef_v4 = lvbo(data(a,i,:),13,14,30,31,12,1,32);
        coef_v5 = detcoef(coe,lis,2);%¦Ã²¨
%        coef_v5 = lvbo(data(a,i,:),3.5,4,7,8,3,1,9);
        temp1(a,i)=mean(coef_v1.^2);
        temp2(a,i)=mean(coef_v2.^2);
        temp3(a,i)=mean(coef_v3.^2);
        temp4(a,i)=mean(coef_v4.^2);
        temp5(a,i)=mean(coef_v5.^2);
        temp6(a,i)=std(coef_v1);
        temp7(a,i)=std(coef_v2);        
        temp8(a,i)=std(coef_v3); 
        temp9(a,i)=std(coef_v4);
        temp10(a,i)=std(coef_v5);
        temp11(a,i)=mean(coef_v1);
        temp12(a,i)=mean(coef_v2);
        temp13(a,i)=mean(coef_v3);
        temp14(a,i)=mean(coef_v4);
        temp15(a,i)=mean(coef_v5);
        temp16(a,i)=max(coef_v1)-min(coef_v1);
        temp17(a,i)=max(coef_v1)-min(coef_v1);        
        temp18(a,i)=max(coef_v1)-min(coef_v1); 
        temp19(a,i)=max(coef_v1)-min(coef_v1);
        temp20(a,i)=max(coef_v1)-min(coef_v1);
    end
 
end

%% ÌØÕ÷
ccc=zeros(num,xindao,20);
srate=160;
tz=zeros(num,xindao,31);
W=zeros(num,xindao,xindao);
w=zeros(num,xindao,xindao);
win=4;

for a=1:num
    temp=[temp1(a,:);temp2(a,:);temp3(a,:);temp4(a,:);temp5(a,:);temp6(a,:);temp8(a,:);temp7(a,:);temp9(a,:);temp10(a,:);
        temp11(a,:);temp12(a,:);temp13(a,:);temp14(a,:);temp15(a,:);temp16(a,:);temp18(a,:);temp17(a,:);temp19(a,:);temp20(a,:)];
    ccc(a,:,:)=temp';
end

for a=1:num
    wave=squeeze(data(a,:,:))';
    [~,DE]=STFT(squeeze(data(a,:,:)),160,[1,4,8,14,30],[4,8,14,30,50],160,win);DE=squeeze(DE)'; %32*5
    power_features = ExtractPowerSpectralFeature(wave, srate);   %6*32
    tz(a,:,:)=[squeeze(ccc(a,:,:)) power_features' DE'];
    W(a,:,:)=goujian(squeeze(tz(a,:,:)),3);
    for c=1:31
        TEMP=squeeze(squeeze(tz(a,:,c)));
        M=max(TEMP);m=min(TEMP);
        TEMP=(TEMP-m)/(M-m);
        tz(a,:,c)=TEMP;
    end
end

%% ÁÚ½ÓÍøÂç
for i=1:num
d=squeeze(W(i,:,:));
A=linjie2(d,1,mean(mean(d))*2/5);
A=A+eye(xindao);
w(i,:,:)=(diag(sum(A)))^(-1/2)*A*(diag(sum(A)))^(-1/2);
end


%%  ÇÐ±ÈÑ©·ò¾í»ý
L_hat=zeros(num,64,64);
for i=1:num
d=squeeze(W(i,:,:));
A=diag(sum(d))-d;
L_hat(i,:,:)=2/eigs(A,1)*A-eye(64);
end
L=zeros(num,3,64,64);
for i=1:num
L(i,1,:,:)=eye(64);
L(i,2,:,:)=L_hat(i,:,:);
L(i,3,:,:)=squeeze(L_hat(i,:,:))*squeeze(L_hat(i,:,:))*2-eye(64);
end

%% ´æ´¢
index = randperm(num);
W = W(index,:,:);
tz = tz(index,:,:);
w = w(index,:,:);
lab = lab(index);
L = L(index,:,:,:);
data=data(index,:,:);
save('feature.mat','W','tz','L','lab');
