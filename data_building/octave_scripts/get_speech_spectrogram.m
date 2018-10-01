# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [flat_S, v_f, v_t] = get_speech_spectrogram (x, FS)

  % Usage: S = get_speech_spectrogram(x, FS)
  %
  % Input:
  %
  % x       - Input Signal
  % FS      - Sampling Frequency
  %
  % Output:
  %
  % flat_S  - Spectrogram as a column vector
  
  debug               = 0;
  n_ms_spectral_slice = 10;
  n_ms_window         = 20;
  n_db_max_clip       = -40;
  n_db_min_clip       = -3;
  n_magnitude         = 4000;
  
  if (n_magnitude > FS/2)
    n_magnitude = FS/2;
  end
  
  step = fix(n_ms_spectral_slice * FS / 1000);  # one spectral slice every 15 ms
  window = fix(n_ms_window * FS/1000);          # 25 ms data window
  fftn = 2^nextpow2(window);                    # next highest power of 2
  
  [S, f, t] = specgram(x, fftn, FS, window, window-step);
  
  S = abs(S(2:fftn*n_magnitude/FS,:));          # magnitude in range 0<f<=4000 Hz.
  S = S/max(S(:));                              # normalize magnitude so that max is 0 dB.
  S = max(S, 10^(n_db_max_clip/10));            # clip below -40 dB.
  S = min(S, 10^(n_db_min_clip/10));            # clip above -3 dB.
  
  if (debug == 1)
    figure();
    imagesc (t, f, log(S));       # display in log scale
    set (gca, "ydir", "normal");  # put the 'y' direction in the correct direction
  end
  
  flat_S = S(:)';
  v_f = f;
  v_t = t;

endfunction
