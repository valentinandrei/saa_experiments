# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_single] = do_ica_speech (v_wavfiles, n_seconds, fs, algo, b_scale)

  addpath("/home/valentin/Working/sw_tools/FastICA");
  addpath("/home/valentin/Working/sw_tools/RadicalICA");
  
  % ----------------------------------------------------------------------------
  % Input
  % ----------------------------------------------------------------------------
  % v_wavfiles  - The vector with wavfile inputs
  % n_seconds   - The selected number of seconds
  % fs          - The sampling frequency
  % b_scale     - If set to 1, scale outputs to (-0.5, 0.5)
  % algo        - ICA algorithm:
  %                 0 - Radical ICA
  %                 1 - Fast ICA
  % ----------------------------------------------------------------------------
  % Output
  % ----------------------------------------------------------------------------
  % m_single    - Single speech sources
  % ----------------------------------------------------------------------------
  
  N       = length(v_wavfiles);
  m_mixed = zeros(N, n_seconds * fs);
  
  % Read wavfiles and do pre-processing
  for i = 1 : N    
    [s, wav_fs, nbits] = wavread(v_wavfiles(i));
    s = s(1 : n_seconds * wav_fs, 1);
    
    if (wav_fs ~= fs)
      printf("Resampling ...\n");
      fflush(stdout);      
      s = resample(s, fs, wav_fs);
    end
    
    m_mixed(i, :) = s;    
  end
  
  % Only RADICAL ICA supported for now  
  [m_single, m_mixing_matrix] = RADICAL(m_mixed);
  
  if (b_scale == 1)
    v_max = max(m_single')';
    m_single = 0.5 .* m_single ./ v_max;    
  end
  
endfunction
