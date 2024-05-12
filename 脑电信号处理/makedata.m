data = zeros(14,12,64,19680);
for j=5:9
    for i=3:9
        b=pop_loadset(strcat('C:\Users\dfsxz\Desktop\高等统计应用\data\去除眼电\S00',num2str(j),'R0',num2str(i),'.set'));
        b=b.data;
        data(j-4,i-2,:,:)=b(1:64,1:19680);
    end
    for i=10:14
        b=pop_loadset(strcat('C:\Users\dfsxz\Desktop\高等统计应用\data\去除眼电\S00',num2str(j),'R',num2str(i),'.set'));
        b=b.data;
        data(j-4,i-2,:,:)=b(1:64,1:19680);
    end
end
for j=10:18
    for i=3:9
        b=pop_loadset(strcat('C:\Users\dfsxz\Desktop\高等统计应用\data\去除眼电\S0',num2str(j),'R0',num2str(i),'.set'));
        b=b.data;
        data(j-4,i-2,:,:)=b(1:64,1:19680);
    end
    for i=10:14
        b=pop_loadset(strcat('C:\Users\dfsxz\Desktop\高等统计应用\data\去除眼电\S0',num2str(j),'R',num2str(i),'.set'));
        b=b.data;
        data(j-4,i-2,:,:)=b(1:64,1:19680);
    end
end