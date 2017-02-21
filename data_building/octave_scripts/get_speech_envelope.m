# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [s_envelope] = get_speech_envelope (s_input, fs, fresample)
  
  s_envelope = abs(hilbert(s_input));
  s_envelope = resample(s_envelope, fresample, fs);
  
endfunction
