% Plot the intermediate representation distribution from original model.

clear all;
close all;

figure('position',[150,100,1000,550]);

RI_1 = [6.1546759605407715, 4.492449760437012, 5.205157279968262,... 
    6.361021995544434, 7.114227294921875, 6.411256313323975,...
    5.557908058166504, 6.074864387512207, 6.649043083190918,...
    5.797237396240234, 6.139648914337158, 6.16121244430542];
RI_2 = [5.6073832511901855, 6.042537212371826, 4.699459075927734,...
    4.753873825073242, 5.801724433898926, 5.377570152282715,...
    5.521845817565918, 4.954897880554199, 6.081811904907227,... 
    5.725761413574219, 6.0917863845825195, 5.3386759757995605];
RI = zeros(12, 2);

for i = 1: length(RI_1)
    RI(i, 1) = RI_1(i);
    RI(i, 2) = RI_2(i);
end

h = bar(RI, 0.8);
set(h(1),'FaceColor', [248,172,140]/255);    
set(h(2),'FaceColor', [147,148,231]/255);

ylabel('Relative Importance','FontSize', 16, 'FontName', 'SansSerif');
xlabel('Channel','FontSize', 16, 'FontName', 'SansSerif');
legend('model 1', 'model 2','FontSize', 13, 'FontName', 'SansSerif');
grid on;