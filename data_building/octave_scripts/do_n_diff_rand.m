# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [v_numbers] = do_n_diff_rand (n, a, b)

  % This function returns n different random numbers between a and b
  
  debug = 0;
  
  if (abs(a - b) < n)
    disp 'Interval is too narrow.';
    break;
  end
  
  v_numbers = zeros(n, 1);
  a = min(a, b);
  range = abs(b - a);
  
  for i = 1 : n
    
    num = randi(range) + a - 1;    
    while (length(find(num == v_numbers)) > 0)
      num = randi(range) + a;
    end
    
    v_numbers(i) = num;
    
  end
  
  if (debug == 1)
    stem(v_numbers);
    grid;
  end

endfunction
