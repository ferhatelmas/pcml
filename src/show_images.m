% display images with y*a_3 closest to zero and largest negative
x_l = X_L_val(:,min_i);
x_r = X_R_val(:,min_i);
x = (x_l + x_r)/2;
x = reshape(x,24,24);
figure(1);imshow(x,'InitialMagnification',500);
title('Fig.1: Most Errored Image (t = 1)');
tmin = T_val(min_i); 

x_l = X_L_val(:,max_i);
x_r = X_R_val(:,max_i);
x = (x_l + x_r)/2;
x = reshape(x,24,24);
figure(2);imshow(x,'InitialMagnification',500);
title('Fig.2: Least Errored Image (t = -1)');
tmax = T_val(max_i);