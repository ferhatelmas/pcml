close all

% display images with y*a_3 closest to zero and largest negative
x_l = test_left_s(:,min_i);
x_r = test_right_s(:,min_i);
x = (x_l + x_r)/2;
x = reshape(uint8(round(x - 1)),24,24);
figure(1);imshow(x,'InitialMagnification',100);
tmin = T_test(min_i); 

x_l = test_left_s(:,max_i);
x_r = test_right_s(:,max_i);
x = (x_l + x_r)/2;
x = reshape(uint8(round(x - 1)),24,24);
figure(2);imshow(uint8(x),'InitialMagnification',100);
tmax = T_test(max_i);