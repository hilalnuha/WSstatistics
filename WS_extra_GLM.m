clc
clear all
close all
%M=csvread("WS_10m.csv",1,3);
%M=csvread("WS_hr.csv",2,3,[2 3 3000 3]);
M=csvread("WS_KFUPM_10m_2015.csv",1,2);
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%M=M(:,[2 4 5 6 7 8 10 

days=360;
%numdat=6*24*days;
inputsize=4;
%M=M(1:(numdat),:);
N=length(M);

for k=1:11
for i=1:N  
    if M(i,k)> 20 %CLEAN THE DATA if 9999, then replace with previous value
        M(i,k)=M(i-1,k);
    elseif M(i,k)<= 0
    M(i,k)=M(i-1,k);
    end
end
end

M=fliplr(M);

ii=1;
for i=1:N
    diff0=M(i,2:11)-M(i,1:10);
    lt0=sum(find(diff0<0.1));
    if lt0==0 
        MN(ii,:)=M(i,:);
        ii=ii+1;
    end
end

mt15=find(MN(:,6)<=15);   
M=MN(mt15,:);
mt10=find(M(:,3)<=10);   
M=M(mt10,:);

M=[M(:,[1 2 3 4]) (M(:,4)+M(:,5))/2 M(:,5) (M(:,5)+M(:,6))/2 M(:,6) (M(:,6)+M(:,7))/2 M(:,7) (M(:,7)+M(:,8))/2 M(:,8) (M(:,8)+M(:,9))/2 M(:,9) (M(:,9)+M(:,10))/2 M(:,10) (M(:,10)+M(:,11))/2 M(:,11)];
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%

perc=100;
numdat=length(M);

%R=6; % Every 6 makes an hour
%mm=floor(N/R);
%for i=1:mm
%    j=(i-1)*R+1;
%    MD(i,1)=mean(M(j:j+R-1));
%end

trainingnum=floor(0.8*numdat); % Num of training samples
maxx=max(max(M(1:trainingnum,1:inputsize)));
training=M(1:trainingnum,:);

series=training/maxx;
datasize=size(series);
nex=1;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing
Nhid=5;
Rrr=0.0000001;
testing=M((trainingnum+1):end,:);

seriesT=testing/maxx;
%numdata=max(datasize)-(inputsize+ahead-1);
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P50 = traininginput';
Y50 = trainingtarget';
Ptest50 = testinginput';
Ytest50 = testingtarget';
testingtarget50=Ytest50'*maxx;
%
%Create NN

%outval = netENS(P);

trainingtargetmax=trainingtarget*maxx;

height50=[10 20 30 40 50];
rang50=[0 13];
rl50=[1:13];
% ENS WSE

ENSP50 = traininginput';
ENSY50 = trainingtarget';
ENSPtest50 = testinginput';
ENSYtest50 = testingtarget';
ENStestingtarget50=ENSYtest50'*maxx;

%netENS = fitglm(ENSP50',ENSY50,'OptimizeHyperparameters','epsilon');
netENS = fitglm(ENSP50',ENSY50);
outval = (predict(netENS,ENSP50'))';

outvalmax=outval*maxx;
ENSOutf50train=outvalmax';
%mse(ENSOutf50train,ENSY50*maxx)
%outvaltest=(sigmoid(Ww*ENSPtest50)'*Beta)';
outvaltest=(predict(netENS,ENSPtest50'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget50;
ENSOutf50=outvaltestmax;
ENSmsetest50=mse(ENSOutf50,testingtarget50);
ENSOut=ENSOutf50;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4);
    ENSmpetest=mpe(ENSOut,testingtargetmax);
    ENSsmapetest=smape(ENSOut,testingtargetmax);
    ENSperf50=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax50=ENSPtest50'*maxx;

meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanENS50=mean([ENSPtestMax50'; ENSOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanENS50,height50,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf50]
perfall=[mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 60

nex=2;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y60 = trainingtarget';
Ytest60 = testingtarget';
testingtarget60=Ytest60'*maxx;

testingtargetmax=testingtarget*maxx;
target60=testingtarget60;

%
height60=[height50 60];
mxr=12.5+nex*0.5;
rang60=[0 mxr];
rl60=[1:mxr];
% ENS WSE
%
ENSP60 = [ENSP50; ENSOutf50train'/maxx];
ENSY60 = trainingtarget';
ENSPtest60 = [ENSPtest50; ENSOutf50'/maxx];
ENSYtest60 = testingtarget';
ENStestingtarget60=ENSYtest60'*maxx;

netENS = fitglm(ENSP60',ENSY60);
outval = (predict(netENS,ENSP60'))';

outvalmax=outval*maxx;
ENSOutf60train=outvalmax';
%mse(ENSOutf60train,ENSY60*maxx)
outvaltest=(predict(netENS,ENSPtest60'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget60;
ENSOut=outvaltestmax;
ENSOutf60=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));
    ENSmpetest=mpe(ENSOut,testingtargetmax);
    ENSsmapetest=smape(ENSOut,testingtargetmax);

    ENSperf60=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax60=ENSPtest60'*maxx;

meantarget60=[meantarget50 mean(testingtarget60)];
meanENS60=[meanENS50 mean(ENSOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanENS60,height60,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf60]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 70

nex=3;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y70 = trainingtarget';
Ytest70 = testingtarget';
testingtarget70=Ytest70'*maxx;

testingtargetmax=testingtarget*maxx;
target70=testingtarget70;

%
height70=[height60 70];
mxr=12.5+nex*0.5;
rang70=[0 mxr];
rl70=[1:mxr];
% ENS WSE
%
ENSP70 = [ENSP60; ENSOutf60train'/maxx];
ENSY70 = trainingtarget';
ENSPtest70 = [ENSPtest60; ENSOutf60'/maxx];
ENSYtest70 = testingtarget';
ENStestingtarget70=ENSYtest70'*maxx;

netENS = fitglm(ENSP70',ENSY70);
outval = (predict(netENS,ENSP70'))';

outvalmax=outval*maxx;
ENSOutf70train=outvalmax';
%mse(ENSOutf70train,ENSY70*maxx)
outvaltest=(predict(netENS,ENSPtest70'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget70;
ENSOut=outvaltestmax;
ENSOutf70=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf70=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax70=ENSPtest70'*maxx;

meantarget70=[meantarget60 mean(testingtarget70)];
meanENS70=[meanENS60 mean(ENSOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanENS70,height70,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf70]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 80

nex=4;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y80 = trainingtarget';
Ytest80 = testingtarget';
testingtarget80=Ytest80'*maxx;

testingtargetmax=testingtarget*maxx;
target80=testingtarget80;

%
height80=[height70 80];
mxr=12.5+nex*0.5;
rang80=[0 mxr];
rl80=[1:mxr];
% ENS WSE
%
ENSP80 = [ENSP70; ENSOutf70train'/maxx];
ENSY80 = trainingtarget';
ENSPtest80 = [ENSPtest70; ENSOutf70'/maxx];
ENSYtest80 = testingtarget';
ENStestingtarget80=ENSYtest80'*maxx;

netENS = fitglm(ENSP80',ENSY80);
outval = (predict(netENS,ENSP80'))';

outvalmax=outval*maxx;
ENSOutf80train=outvalmax';
%mse(ENSOutf80train,ENSY80*maxx)
outvaltest=(predict(netENS,ENSPtest80'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget80;
ENSOut=outvaltestmax;
ENSOutf80=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf80=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax80=ENSPtest80'*maxx;

meantarget80=[meantarget70 mean(testingtarget80)];
meanENS80=[meanENS70 mean(ENSOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanENS80,height80,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf80]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 90

nex=5;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y90 = trainingtarget';
Ytest90 = testingtarget';
testingtarget90=Ytest90'*maxx;

testingtargetmax=testingtarget*maxx;
target90=testingtarget90;

%
height90=[height80 90];
mxr=12.5+nex*0.5;
rang90=[0 mxr];
rl90=[1:mxr];
% ENS WSE
%
ENSP90 = [ENSP80; ENSOutf80train'/maxx];
ENSY90 = trainingtarget';
ENSPtest90 = [ENSPtest80; ENSOutf80'/maxx];
ENSYtest90 = testingtarget';
ENStestingtarget90=ENSYtest90'*maxx;

netENS = fitglm(ENSP90',ENSY90);
outval = (predict(netENS,ENSP90'))';

outvalmax=outval*maxx;
ENSOutf90train=outvalmax';
%mse(ENSOutf90train,ENSY90*maxx)
outvaltest=(predict(netENS,ENSPtest90'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget90;
ENSOut=outvaltestmax;
ENSOutf90=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf90=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax90=ENSPtest90'*maxx;

meantarget90=[meantarget80 mean(testingtarget90)];
meanENS90=[meanENS80 mean(ENSOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanENS90,height90,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf90]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 100

nex=6;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y100 = trainingtarget';
Ytest100 = testingtarget';
testingtarget100=Ytest100'*maxx;

testingtargetmax=testingtarget*maxx;
target100=testingtarget100;

%
height100=[height90 100];
mxr=12.5+nex*0.5;
rang100=[0 mxr];
rl100=[1:mxr];
% ENS WSE
%
ENSP100 = [ENSP90; ENSOutf90train'/maxx];
ENSY100 = trainingtarget';
ENSPtest100 = [ENSPtest90; ENSOutf90'/maxx];
ENSYtest100 = testingtarget';
ENStestingtarget100=ENSYtest100'*maxx;

netENS = fitglm(ENSP100',ENSY100);
outval = (predict(netENS,ENSP100'))';

outvalmax=outval*maxx;
ENSOutf100train=outvalmax';
%mse(ENSOutf100train,ENSY100*maxx)
outvaltest=(predict(netENS,ENSPtest100'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget100;
ENSOut=outvaltestmax;
ENSOutf100=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf100=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax100=ENSPtest100'*maxx;

meantarget100=[meantarget90 mean(testingtarget100)];
meanENS100=[meanENS90 mean(ENSOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanENS100,height100,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf100]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 110

nex=7;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y110 = trainingtarget';
Ytest110 = testingtarget';
testingtarget110=Ytest110'*maxx;

testingtargetmax=testingtarget*maxx;
target110=testingtarget110;

%
height110=[height100 110];
mxr=12.5+nex*0.5;
rang110=[0 mxr];
rl110=[1:mxr];
% ENS WSE
%
ENSP110 = [ENSP100; ENSOutf100train'/maxx];
ENSY110 = trainingtarget';
ENSPtest110 = [ENSPtest100; ENSOutf100'/maxx];
ENSYtest110 = testingtarget';
ENStestingtarget110=ENSYtest110'*maxx;

netENS = fitglm(ENSP110',ENSY110);
outval = (predict(netENS,ENSP110'))';

outvalmax=outval*maxx;
ENSOutf110train=outvalmax';
%mse(ENSOutf110train,ENSY110*maxx)
outvaltest=(predict(netENS,ENSPtest110'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget110;
ENSOut=outvaltestmax;
ENSOutf110=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf110=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax110=ENSPtest110'*maxx;

meantarget110=[meantarget100 mean(testingtarget110)];
meanENS110=[meanENS100 mean(ENSOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanENS110,height110,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf110]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 120

nex=8;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y120 = trainingtarget';
Ytest120 = testingtarget';
testingtarget120=Ytest120'*maxx;

testingtargetmax=testingtarget*maxx;
target120=testingtarget120;

%
height120=[height110 120];
mxr=12.5+nex*0.5;
rang120=[0 mxr];
rl120=[1:mxr];
% ENS WSE
%
ENSP120 = [ENSP110; ENSOutf110train'/maxx];
ENSY120 = trainingtarget';
ENSPtest120 = [ENSPtest110; ENSOutf110'/maxx];
ENSYtest120 = testingtarget';
ENStestingtarget120=ENSYtest120'*maxx;

netENS = fitglm(ENSP120',ENSY120);
outval = (predict(netENS,ENSP120'))';

outvalmax=outval*maxx;
ENSOutf120train=outvalmax';
%mse(ENSOutf120train,ENSY120*maxx)
outvaltest=(predict(netENS,ENSPtest120'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget120;
ENSOut=outvaltestmax;
ENSOutf120=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf120=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax120=ENSPtest120'*maxx;

meantarget120=[meantarget110 mean(testingtarget120)];
meanENS120=[meanENS110 mean(ENSOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanENS120,height120,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf120]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%%
%% 130

nex=9;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y130 = trainingtarget';
Ytest130 = testingtarget';
testingtarget130=Ytest130'*maxx;

testingtargetmax=testingtarget*maxx;
target130=testingtarget130;

%
height130=[height120 130];
mxr=12.5+nex*0.5;
rang130=[0 mxr];
rl130=[1:mxr];
% ENS WSE
%
ENSP130 = [ENSP120; ENSOutf120train'/maxx];
ENSY130 = trainingtarget';
ENSPtest130 = [ENSPtest120; ENSOutf120'/maxx];
ENSYtest130 = testingtarget';
ENStestingtarget130=ENSYtest130'*maxx;

netENS = fitglm(ENSP130',ENSY130);
outval = (predict(netENS,ENSP130'))';

outvalmax=outval*maxx;
ENSOutf130train=outvalmax';
%mse(ENSOutf130train,ENSY130*maxx)
outvaltest=(predict(netENS,ENSPtest130'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget130;
ENSOut=outvaltestmax;
ENSOutf130=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf130=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax130=ENSPtest130'*maxx;

meantarget130=[meantarget120 mean(testingtarget130)];
meanENS130=[meanENS120 mean(ENSOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanENS130,height130,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf130]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 140

nex=10;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y140 = trainingtarget';
Ytest140 = testingtarget';
testingtarget140=Ytest140'*maxx;

testingtargetmax=testingtarget*maxx;
target140=testingtarget140;

%
height140=[height130 140];
mxr=12.5+nex*0.5;
rang140=[0 mxr];
rl140=[1:mxr];
% ENS WSE
%
ENSP140 = [ENSP130; ENSOutf130train'/maxx];
ENSY140 = trainingtarget';
ENSPtest140 = [ENSPtest130; ENSOutf130'/maxx];
ENSYtest140 = testingtarget';
ENStestingtarget140=ENSYtest140'*maxx;

netENS = fitglm(ENSP140',ENSY140);
outval = (predict(netENS,ENSP140'))';

outvalmax=outval*maxx;
ENSOutf140train=outvalmax';
%mse(ENSOutf140train,ENSY140*maxx)
outvaltest=(predict(netENS,ENSPtest140'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget140;
ENSOut=outvaltestmax;
ENSOutf140=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf140=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax140=ENSPtest140'*maxx;

meantarget140=[meantarget130 mean(testingtarget140)];
meanENS140=[meanENS130 mean(ENSOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanENS140,height140,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf140]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];


%% 150

nex=11;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y150 = trainingtarget';
Ytest150 = testingtarget';
testingtarget150=Ytest150'*maxx;

testingtargetmax=testingtarget*maxx;
target150=testingtarget150;

%
height150=[height140 150];
mxr=12.5+nex*0.5;
rang150=[0 mxr];
rl150=[1:mxr];
% ENS WSE
%
ENSP150 = [ENSP140; ENSOutf140train'/maxx];
ENSY150 = trainingtarget';
ENSPtest150 = [ENSPtest140; ENSOutf140'/maxx];
ENSYtest150 = testingtarget';
ENStestingtarget150=ENSYtest150'*maxx;

netENS = fitglm(ENSP150',ENSY150);
outval = (predict(netENS,ENSP150'))';

outvalmax=outval*maxx;
ENSOutf150train=outvalmax';
%mse(ENSOutf150train,ENSY150*maxx)
outvaltest=(predict(netENS,ENSPtest150'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget150;
ENSOut=outvaltestmax;
ENSOutf150=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf150=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax150=ENSPtest150'*maxx;

meantarget150=[meantarget140 mean(testingtarget150)];
meanENS150=[meanENS140 mean(ENSOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanENS150,height150,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf150]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 160

nex=12;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y160 = trainingtarget';
Ytest160 = testingtarget';
testingtarget160=Ytest160'*maxx;

testingtargetmax=testingtarget*maxx;
target160=testingtarget160;

%
height160=[height150 160];
mxr=12.5+nex*0.5;
rang160=[0 mxr];
rl160=[1:mxr];
% ENS WSE
%
ENSP160 = [ENSP150; ENSOutf150train'/maxx];
ENSY160 = trainingtarget';
ENSPtest160 = [ENSPtest150; ENSOutf150'/maxx];
ENSYtest160 = testingtarget';
ENStestingtarget160=ENSYtest160'*maxx;

netENS = fitglm(ENSP160',ENSY160);
outval = (predict(netENS,ENSP160'))';

outvalmax=outval*maxx;
ENSOutf160train=outvalmax';
%mse(ENSOutf160train,ENSY160*maxx)
outvaltest=(predict(netENS,ENSPtest160'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget160;
ENSOut=outvaltestmax;
ENSOutf160=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf160=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax160=ENSPtest160'*maxx;

meantarget160=[meantarget150 mean(testingtarget160)];
meanENS160=[meanENS150 mean(ENSOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanENS160,height160,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf160]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
%% 170

nex=13;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y170 = trainingtarget';
Ytest170 = testingtarget';
testingtarget170=Ytest170'*maxx;

testingtargetmax=testingtarget*maxx;
target170=testingtarget170;

%
height170=[height160 170];
mxr=12.5+nex*0.5;
rang170=[0 mxr];
rl170=[1:mxr];
% ENS WSE
%
ENSP170 = [ENSP160; ENSOutf160train'/maxx];
ENSY170 = trainingtarget';
ENSPtest170 = [ENSPtest160; ENSOutf160'/maxx];
ENSYtest170 = testingtarget';
ENStestingtarget170=ENSYtest170'*maxx;

netENS = fitglm(ENSP170',ENSY170);
outval = (predict(netENS,ENSP170'))';

outvalmax=outval*maxx;
ENSOutf170train=outvalmax';
%mse(ENSOutf170train,ENSY170*maxx)
outvaltest=(predict(netENS,ENSPtest170'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget170;
ENSOut=outvaltestmax;
ENSOutf170=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf170=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax170=ENSPtest170'*maxx;

meantarget170=[meantarget160 mean(testingtarget170)];
meanENS170=[meanENS160 mean(ENSOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanENS170,height170,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf170]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 180

nex=14;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y180 = trainingtarget';
Ytest180 = testingtarget';
testingtarget180=Ytest180'*maxx;

testingtargetmax=testingtarget*maxx;
target180=testingtarget180;

%
height180=[height170 180];
mxr=12.5+nex*0.5;
rang180=[0 mxr];
rl180=[1:mxr];
% ENS WSE
%
ENSP180 = [ENSP170; ENSOutf170train'/maxx];
ENSY180 = trainingtarget';
ENSPtest180 = [ENSPtest170; ENSOutf170'/maxx];
ENSYtest180 = testingtarget';
ENStestingtarget180=ENSYtest180'*maxx;

netENS = fitglm(ENSP180',ENSY180);
outval = (predict(netENS,ENSP180'))';

outvalmax=outval*maxx;
ENSOutf180train=outvalmax';
%mse(ENSOutf180train,ENSY180*maxx)
outvaltest=(predict(netENS,ENSPtest180'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget180;
ENSOut=outvaltestmax;
ENSOutf180=ENSOut;
    ENSmsetest=mse(ENSOut,testingtargetmax);
    ENSrmsetest=rmse(ENSOut,testingtargetmax);
    ENSmaetest=mae(ENSOut,testingtargetmax);
    ENSmbetest=mbe(ENSOut,testingtargetmax);
    ENSnmsetest=nmse(ENSOut,testingtargetmax);
    ENSnrmsetest=nrmse(ENSOut,testingtargetmax);
    ENSmapetest=mape(ENSOut,testingtargetmax);
    ENSr2test=rsquare(ENSOut,testingtargetmax);
    ENSadjr2=adjr2(ENSOut,testingtargetmax,4+(nex-1));     ENSmpetest=mpe(ENSOut,testingtargetmax);     ENSsmapetest=smape(ENSOut,testingtargetmax); 
    ENSperf180=[ENSmsetest ENSrmsetest ENSmaetest ENSmbetest ENSnmsetest ENSnrmsetest ENSmapetest*100 ENSr2test*100 ENSadjr2*100 ENSmpetest ENSsmapetest];
ENSPtestMax180=ENSPtest180'*maxx;

meantarget180=[meantarget170 mean(testingtarget180)];
meanENS180=[meanENS170 mean(ENSOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanENS180,height180,'-.g');

hold off
title('average')
legend('measured','ENS est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
[ENSperf180]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];


%%
ENSPerfAll=[ENSperf50;ENSperf60;ENSperf70;ENSperf80;ENSperf90;ENSperf100;ENSperf110;ENSperf120; ENSperf130; ENSperf140; ENSperf150; ENSperf160; ENSperf170; ; ENSperf180];
