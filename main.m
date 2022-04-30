clc
clear
close all
%% author farhad abedinzadeh


load('BCICIV_calib_ds1a_100Hz.mat')
cnt= double(cnt)*0.1; % uv
ypos= nfo.ypos;
xpos= nfo.xpos;
plot(xpos,ypos,'ob','linewidth',2,'markersize',10)
grid on
grid minor
name= nfo.clab;
text(xpos+0.02,ypos+0.02,name,'fontsize',10)

%%
pos= mrk.pos;
label= mrk.y;
c1=0;
c2=0;
for i=1:numel(label)
    indx= pos(i):pos(i)+399;
    single_trial= cnt(indx,:);
    if label(i)==1
        c1=c1+1;
        data1(:,:,c1)= single_trial;
    elseif label(i)== -1
        c2=c2+1;
        data2(:,:,c2)= single_trial;
    end
    
end



Fs=nfo.fs;
%% step 1: source localization
% car - spatial filter
ref= mean(cnt,2);
for i=1:size(cnt,2)
    cnt(:,i)= cnt(:,i)-ref;
end
%% step 2:  band pass filtering to get beta and mu band information
% butter-bandpass filter(8-30Hz)
[b,a]= butter(3,[8 30]/(Fs/2),'bandpass');
cnt= filtfilt(b,a,cnt);
%% step 3: channel selection
% 

%% step 4:  cut trials from eeg
pos_tr= mrk.pos;
label= mrk.y;
c1=0;
c2=0;
for i=1:numel(label)
    ind= pos_tr(i):pos_tr(i)+399;
    sigleTrail = cnt(ind,:);
    if label(i) ==1
        c1=c1+1;
        data1(:,:,c1)=sigleTrail;
    elseif label(i)== -1
        c2=c2+1;
        data2(:,:,c2)=sigleTrail;
    end
end

%% step 5: feature extaction
for i=1:size(data1,3)
    X1= data1(:,:,i);
    X2= data2(:,:,i);
    for j=1:size(X1,2)
        tp1(:,j)= myfeatureExtraction(X1(:,j));
        tp2(:,j)= myfeatureExtraction(X2(:,j));
    end
    Features1(:,i)= tp1(:);
    Features2(:,i)= tp2(:);
end

%% step 6: classification
% step 6-1: devide data into train and test data
% k-fold cross validation(k=5)
k=5;
fold1=floor(size(Features1,2)/k);
fold2=floor(size(Features2,2)/k);
Ct=0;
for iter=1:k
    %% training and test
    % class 1
    testindex1=(iter-1)*fold1+1:(iter-1)*fold1+fold1;
    trainindex1=1:size(Features1,2);
    trainindex1(testindex1)=[];
    traindata1=Features1(:,trainindex1);
    testdata1=Features1(:,testindex1);
    % class 2
    testindex2=(iter-1)*fold2+1:(iter-1)*fold2+fold2;
    
    trainindex2=1:size(Features2,2);
    trainindex2(testindex2)=[];
    traindata2=Features2(:,trainindex2);
    testdata2=Features2(:,testindex2);

    labeltrain=[ones(1,size(traindata1,2)),2*ones(1,size(traindata2,2))];
    traindata=[traindata1,traindata2];
    
    testdata=[testdata1,testdata2];
    testlabel=[ones(1,size(testdata1,2)),2*ones(1,size(testdata2,2))];
    %%  step 6-2: train model using train data and train label
    mdl=fitcsvm(traindata',labeltrain,'Standardize',1);
    %% step 6-3: test trained model using test data
    output =predict(mdl,testdata');
    %% step 6-4: validation
    C= confusionmat(testlabel,output);
    Ct= Ct+C;
    accuracy(iter)= sum(diag(C)) / sum(C(:))*100;
    sensitivity(iter)= C(1,1) / sum(C(1,:))*100;
    specificity(iter)= C(2,2) / sum(C(2,:))*100;
end
Ct
disp(['Total Accuracy: ',num2str(mean(accuracy)),'%'])
disp(['Sensitivity: ',num2str(mean(sensitivity)),'%'])
disp(['Specificity: ',num2str(mean(specificity)),'%'])



