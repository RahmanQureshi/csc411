y = linspace(-100,100,1000);
t = 0;
delta = 50;
Loriginal = 0.5*(y-t).^2;
Lhuber = huberloss(y,t,delta);

figure
hold on
plot(y, Loriginal);
plot(y, Lhuber);
