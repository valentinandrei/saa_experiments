# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [s_fft] = get_speech_spectrum (s_input, fs, fstop)

  s_fft = abs(fft(s_input));
  N = ceil((length(s_fft)/2) * fstop / fs);  
  s_fft = s_fft(1 : N);
  
endfunction