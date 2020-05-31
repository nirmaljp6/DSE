%This file performs data preprocessing of the deep state estimation (DSE)
%method of Nair and Goza, 2020
%Preprocessing includes gathering all snapshots, 
%computing POD modes and segregating the
%snapshots into training, validation and testing datasets

%Warning: Extracting snapshots can sometimes cause 'out of memory' issues 

clc; clear all;
%User inputs------------------------------------------------------------
%Select case
aoa = 70; %choose 25 or 70 deg
k = 25; %Number of POD modes
s = 5; %Number of sensors

%Select which snapshots to include for computing POD modes
select_phi = 2; %1: vorticity, 2: vorticity and surface stress
%Select which sensor 
sense = 2; %1: vorticity,  2: surface stress magnitude
%End of inputs---------------------------------------------------------

%Setup parametric variations
if aoa==25
    nsnaps = 250; nx=499; ny=399; ds = 0.01; aoa = 25; offsetx=1; offsety=2; 
    param_aoa = 25:0.2:27;
    param_aoa_predict = [25.5,26.25,26.75];
elseif aoa==70
    nsnaps = 400; nx=399; ny=599; ds = 0.01; aoa = 70; offsetx=1; offsety=1; 
    param_aoa = 70:0.2:71;
    param_aoa_predict = [70.25,70.5,70.75];
end

%Extracting snapshots------------------------------------------------
X=[]; Xstress=[];
shuffle_train = []; shuffle_valid = []; shuffle_test = [];
for i=1:length(param_aoa)
    filename = ['../snapshots/snapshot_data' num2str(param_aoa(i),'%2.2f') '.mat'];
    Y=load(filename);  %loads snapshots in X
    X = [X, Y.X(:,1:nsnaps)];
    m=size(Y.X(:,1:nsnaps),2);   %number of snapshots
    shuffle = size(X,2) - m + randperm(m);
    filename = ['../snapshots/stress_data' num2str(param_aoa(i),'%2.2f') '.mat'];
    Y = load(filename);  %loads snapshot matrix in Xstress
    Xstress = [Xstress, Y.Xstress(:,1:nsnaps)];
    
    train_size = floor(0.8*m); %80-20% split of training and validation snapshots
    valid_size = m-train_size;
    shuffle_train = [shuffle_train, shuffle(1:train_size)];
    shuffle_valid = [shuffle_valid, shuffle(train_size+1:end)];
    clear Y
end

[Uhat, mean_X, mean_Xstress] = POD(X, Xstress, select_phi);

test_size = floor(train_size/0.8 - train_size);
for i=1:length(param_aoa_predict)
    
    filename = ['../snapshots/snapshot_data' num2str(param_aoa_predict(i),'%2.2f') '.mat'];
    Y=load(filename);  %loads snapshot matrix in X
    Y.X = Y.X(:,1:nsnaps);
    m=size(Y.X,2);   %number of snapshots
    shuffle = randperm(m);
    X = [X, Y.X(:,shuffle(1:test_size))];
    filename = ['../snapshots/stress_data' num2str(param_aoa_predict(i),'%2.2f') '.mat'];
    Y=load(filename);  %loads snapshot matrix in Xstress
    Xstress = [Xstress, Y.Xstress(:,shuffle(1:test_size))];
    
    shuffle_test = [shuffle_test, size(X,2)-test_size+1:size(X,2)];
    clear Y
end

load('../snapshots/body_data_maoa.mat')   %loads the matrix Xb
%Transforming surface stress snapshots
nb = size(Xstress,1)/2;
Xstress =  (Xstress(1:nb,:).^2 + Xstress(nb+1:2*nb,:).^2).^0.5;
%Combining vorticity and surface stress snapshots
if select_phi==2
   X = [X;Xstress];
   mean_X = [mean_X; mean_Xstress];
end
phi = Uhat(:,1:k); clear Uhat

%Sensor locations
si = 10; se = 90;  
Z_snew = Xb(floor(linspace(si,se,s)),1:2)';
ind = Z_snew/ds;
ind(1,:) = ind(1,:) + (offsetx/ds);
ind(2,:) = ind(2,:) + (offsety/ds);
if sense==1
    sensor_loc = round(ny*(round(ind(1,:))-1) + round(ind(2,:)));
elseif sense==2
    imax=size(X,1);
    sensor_loc = imax + floor(linspace(si,se,s));
end

%Collecting sensor data
if sense==1
    sensor_dat = X(sensor_loc(:),:);
    sensor_dat = sensor_dat';
elseif sense==2
    sensor_dat = Xstress(floor(linspace(si,se,s)),:);
    sensor_dat = sensor_dat';
end

%Collecting generalized coordinates by projection
if select_phi==1
    gc_dat = phi'*(bsxfun(@minus,X,mean_X));
elseif select_phi==2
    gc_dat = phi'*(bsxfun(@minus,X,mean_X));
end
gc_dat = gc_dat';
state_dat = X';

clear X Xstress;

%Sampling training, validation and testing datasets
train_dat_in = sensor_dat(shuffle_train(:),:);
train_dat_out = gc_dat(shuffle_train(:),:);
valid_dat_in = sensor_dat(shuffle_valid(:),:);
valid_dat_out = gc_dat(shuffle_valid(:),:);
test_dat_in = sensor_dat(shuffle_test(:),:);
test_dat_out = gc_dat(shuffle_test(:),:);
state_dat = state_dat(shuffle_test(:),:);

%Standardization-------
stats_in(1,:)=mean(train_dat_in,1);
stats_in(2,:)=std(train_dat_in,1);
stats_out(1,:)=mean(train_dat_out,1);
stats_out(2,:)=std(train_dat_out,1);

%Saving sanpshots
dlmwrite('data/train_in.csv',train_dat_in,'delimiter', ',', 'precision', 9);
dlmwrite('data/train_out.csv',train_dat_out,'delimiter', ',', 'precision', 9);
dlmwrite('data/valid_in.csv',valid_dat_in,'delimiter', ',', 'precision', 9);
dlmwrite('data/valid_out.csv',valid_dat_out,'delimiter', ',', 'precision', 9);
dlmwrite('data/test_in.csv',test_dat_in,'delimiter', ',', 'precision', 9);
dlmwrite('data/test_out.csv',test_dat_out,'delimiter', ',', 'precision', 9);
dlmwrite('data/state.csv',state_dat,'delimiter', ',', 'precision', 9); save('data/state.mat','state_dat','-v7.3');
dlmwrite('data/state_mean.csv',mean_X','delimiter', ',', 'precision', 9);
dlmwrite('data/basis.csv',phi,'delimiter', ',', 'precision', 9);
dlmwrite('data/sensor_loc.csv',sensor_loc,'delimiter', ',', 'precision', 9);
dlmwrite('data/sensor_index.csv',ind,'delimiter', ',', 'precision', 9);
dlmwrite('data/stats_in.csv',stats_in,'delimiter', ',', 'precision', 9);
dlmwrite('data/stats_out.csv',stats_out,'delimiter', ',', 'precision', 9);
