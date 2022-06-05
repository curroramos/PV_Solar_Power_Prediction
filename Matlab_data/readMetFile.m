function [vt,vd] = readMetFile
close all
zonaHoraria = 'UTC';
directorio='/Users/curro/PV_prediction/Matlab_data'; % Directorio con los ficheros meteorológicos. HAY QUE PONER DOBLE BARRA \\
fechaHoraInicio    = datetime([2014 12 25  0  0 0],'TimeZone',zonaHoraria);
fechaHoraFin       = datetime([2014 12 30 23 59 55],'TimeZone',zonaHoraria);
intervalo   = duration(0,5,0);
colData=[13;... % Radiación directa (W/m2)
    16;...% Temperatura (ºC)
    19;...% Presión barométrica (MBar)
    20];  % Humedad relativa
colDesc={'Irradiancia solar.  NIP pyrheliometer (W/m2)',...
    'Temperatura (ºC)',...
    'Presión barométrica (MBar)',...
    'Humedad Relativa (%)',...
    'Irradiancia Teórica (W/m2)',...
    'Desviación Radiación directa (W/m2)'};

% Datos de la planta solar
simInit.PVpanelSurface = 1;           % (m^2)  - Superficie de paneles solares
simInit.PVlatitud  = 37.41136;          % (º)    - Latitud de la ubicación de la planta solar
simInit.PVlongitud = -6.00055;          % (º)    - Longitud de la ubicación de la planta solar
% Parámetros de lo módulos
simInit.PVtheta = simInit.PVlatitud;    % (º)    - theta: grados de inclinación de las placas con respecto al suelo (Lo tomamos igual a la latitud)
simInit.PVGref = 1000; %                % (W/m2) - Gref: Radiación solar de referencia
simInit.PVnominalPower = 800e3;         % (W)    - PV_nominal: Potencia nominal de la planta solar instalada
simInit.PVPw_stc = 1000;                % (W)    - Ppv_stc : Potencia nominal (rated power) del panel bajo condiciones de test standard
simInit.PVeta = 0.85;                   % (%)    - eta_pv: Eficiencia del panel
simInit.PVTc_stc = 25;                  % (ºC)   - Tc_stc: Referencia de temperatura de la celda bajo condiciones de test standard
simInit.PVbeta = 0.0045;                % (º)    - beta: coeficiente de temperatura [0.004 0.006] per ºC de celda
simInit.Tamb = 25;                      % (ºC)   - Tamb: Temperatura ambiente
simInit.PVNOCT = 55;                    % (ºC)   - NOCT: Normal Operation Cell Temperature

t1 = fechaHoraInicio;
k=1;
vt=[];
vd=[];
x=[];

M=leeMETfecha(directorio,t1);
y1 = ones(length(M{1}),1) * [year(t1) month(t1) day(t1)];
vTime=datetime([ y1 M{1} M{2} M{3}],'TimeZone',zonaHoraria);
vData=[];
for i=1:length(colData)
    vData = [vData M{colData(i)}];
end

while t1<fechaHoraFin
    currentFileDate = 366*year(t1)+day(t1,'dayofyear');
    t2=t1+intervalo;
    nextFileDate = 366*year(t2)+day(t2,'dayofyear');
    if currentFileDate ~= nextFileDate
        Mnext = leeMETfecha(directorio,t2);
        y1 = ones(length(M{1}),1) * [year(t1) month(t1) day(t1)];
        ynext = ones(length(Mnext{1}),1) * [year(t2) month(t2) day(t2)];
        vTime=datetime([ y1 M{1} M{2} M{3} ; ynext Mnext{1} Mnext{2} Mnext{3}],'TimeZone',zonaHoraria);
        vData=[];
        for i=1:length(colData)
            vData = [vData [M{colData(i)};Mnext{colData(i)}]];
        end
    else
        Mnext = M;
    end
    % Datos de radiación teórica
    x = PVPower(simInit,x,t1,k);
    
    
    indInterval = logical((vTime<t2)-(vTime<=t1));
    aux = mean(vData(indInterval,:));
    radDirTeor = x.PVPower(k)*1.40;
    vd=[vd; [aux  radDirTeor aux(1)-radDirTeor ]];
    vt=[vt; t1];
    
    t1=t2;
    k=k+1;
    M = Mnext;
end
%size(vd)
%size(vt)
for i=1:size(vd,2)
    figure
    plot(vt,vd(:,i));
    ylabel(colDesc{i});
end
figure
plot(vt,vd(:,1),vt,vd(:,5));
ylabel('Solar Irradiance vs measured (W/m2)');
save('solarRad.mat','vt','vd')
% Coeficiente de ajuste Min cuad
Pt = vd(:,5);
Pr = vd(:,1);
alpha = Pt'*Pr/(Pt'*Pt)
max(Pr)/max(Pt)
end

function M=leeMETfecha(directorio,fecha)
fileName = sprintf('%s/%d/meteo_%d_%03d.txt',directorio,year(fecha), year(fecha),day(fecha,'dayofyear')); %Modificacion MAC
fprintf('Abriendo fichero\n%s\n',fileName);
fileID=fopen(fileName);
M=textscan(fileID,['%d:%d:%d' repmat('%f',1,18)], ...
    'delimiter',' ', ...
    'headerlines',0, ...
    'multipledelimsasone',1);
fclose(fileID);
end

function x = PVPower(simInit,x,t,k)
% Posición del sol
[declin,~,sol_lon] = EarthEphemeris(t);
[mu0,phi0] = sunang(simInit.PVlatitud, simInit.PVlongitud,declin,sol_lon);
mu = sunslope(mu0,phi0,0,0);
% Elevación angular del sol desde el horizonte en grados
elevacion = 90-acosd(mu);
% Orientación del panel
azimuth = phi0;
if simInit.PVlatitud>0 % Si estamos en el hemisferio norte
    Phi = 180; % Orientación del sol respecto al sur geográfico (anti-clockwise) = 180-azimuth
else
    Phi = 0;   % Apuntamos al norte
end

% Potencia extraida de paneles
%Gt(t): Radiación solar perpendicular al panel fotovoltaico
Gt = simInit.PVGref*(cosd(elevacion)*sind(simInit.PVtheta)*cosd(Phi-azimuth)+sind(elevacion)*cosd(simInit.PVtheta));
%Tc: Temperatura de las celdas del panel fotovoltaico
x.PVTc(k) = simInit.Tamb + (simInit.PVNOCT-20)*Gt/800;
%Ppv: Potencia extraida de los paneles
Ppv = (Gt/simInit.PVGref)*simInit.PVPw_stc*simInit.PVeta*(1-simInit.PVbeta*(x.PVTc(k)-simInit.PVTc_stc));
% Si se obtiene un valor negativo, la potencia extraida debe ser 0
x.PVPower(k) = simInit.PVpanelSurface * max([0 Ppv]);

end
