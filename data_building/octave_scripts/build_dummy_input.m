n_features  = 500;
n_classes   = 4;
n_samples   = 10000;

% Generate random numbers for X that will be modified
X = rand(n_samples, n_features);
Y = zeros(n_samples, n_classes);

for i = 1 : n_samples
  class = randi(n_classes);
  Y(i, class) = 1;
  
  for j = 1 : n_features
    if (mod(j, class + 1) == 0)
      temp = 0;
      for k = j : -1 : (j - class)
        temp = temp + X(i, k);
      endfor
      X(i, j) = temp / class;
    endif
  endfor
endfor

save("-ascii", "x_dummy.txt", "X");
save("-ascii", "y_dummy.txt", "Y");