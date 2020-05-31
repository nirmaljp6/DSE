function [Uhat, mean_X, mean_Xstress] = POD(Xtil, Xtil2, select_phi)

%Subtract mean from vorticity snapshots
mean_X = mean( Xtil, 2 );
for j = 1 : size(Xtil,2)
    Xtil(:,j) = Xtil(:,j) - mean_X;
end
%Transforming surface stress snapshots
nb = size(Xtil2,1)/2;
Xtil2 =  (Xtil2(1:nb,:).^2 + Xtil2(nb+1:2*nb,:).^2).^0.5;
%Subtract mean from surface stress snapshots
mean_Xstress = mean( Xtil2, 2 );
for j = 1 : size(Xtil2,2)
    Xtil2(:,j) = Xtil2(:,j) - mean_Xstress;
end

if select_phi==2
    Xtil = [Xtil;Xtil2];
end

%--get singular vectors V
smallmat = (Xtil' * Xtil );
[V,Sig_sq] = eig( smallmat );

%sort sing vals in order of decreasing magnitude
[Sigsq, ind_srt] = sort( diag( Sig_sq ), 'descend' );
V = V(:, ind_srt );

Sig = real(sqrt( Sigsq ));
Siginv = Sig;
Siginv( Siginv ~= 0 ) = 1./ Siginv( Siginv ~= 0 );
Siginv = diag(Siginv);

Uhat = Xtil * V; clear Xtil; 
Uhat = Uhat * Siginv;
Uhat = Uhat(:,1:50);