para = [500 240 100 40;500 240 100 80; 500 240 100 120; 500 240 100 160; 500 240 80 80; 500 240 100 80; 500 240 120 80; 500 240 140 80; 500 200 100 80; 500 240 100 80; 500 280 100 80; 500 320 100 80; 400 240 100 80; 500 240 100 80;  600 240 100 80;  700 240 100 80];

m =size(para,1);
% m = 6;
refls = zeros(m, 81);
acc=10;
show1=1;
wave = 380:5:780;
stepcase =5;

tic
for i=1:1:m
    refls(i,:)=RCWA_Silicon(para(i,4),para(i,2),para(i,1),para(i,3),acc, show1, stepcase);
    i
end
T=toc

figure(1)
subplot(2,2,4)
plot(wave, refls(1,:),wave,refls(2,:),wave,refls(3,:), wave, refls(4,:))
legend({'40nm','80nm','120nm','160nm'});
axis([380 780 0 0.5]);
xlabel('Wavelength/(nm)');
ylabel('Reflection');

subplot(2,2,3)
plot(wave, refls(5,:),wave,refls(6,:),wave,refls(7,:), wave, refls(8,:))
legend({'80nm','100nm','120nm','140nm'});
axis([380 780 0 0.4]);
xlabel('Wavelength/(nm)');
ylabel('Reflection');

subplot(2,2,2)
plot(wave, refls(9,:),wave,refls(10,:),wave,refls(11,:), wave, refls(12,:))
legend({'200nm','240nm','280nm','320nm'});
axis([380 780 0 0.3]);
xlabel('Wavelength/(nm)');
ylabel('Reflection');


subplot(2,2,1)
plot(wave, refls(13,:),wave,refls(14,:),wave,refls(15,:), wave, refls(16,:))
legend({'400nm','500nm','600nm','700nm'});
axis([380 780 0 0.5]);
xlabel('Wavelength/(nm)');
ylabel('Reflection');


% tStart = tic; 
% T = zeros(m,m);
% for i = 1:1:m
%     for j = 1:1:1
%         acc = 5*i;
%         stepcase = 5*j;
%         tic  
%         refls((i-1)*m+j,:)=RCWA_Silicon(para(4,4),para(4,2),para(4,1),para(4,3),acc, show1,stepcase);
%         T(i,j)= toc;
%         (i-1)*m+j
%     end
% end

