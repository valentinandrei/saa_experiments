# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_features] = build_features (m_mixtures, fs, frame_ms, v_features)

  % ----------------------------------------------------------------------------
  % Input:
  % ----------------------------------------------------------------------------
  % m_mixtures  - the matrix with speech mixtures
  % fs          - targeted sampling frequency in Hz
  % frame_ms    - the number of milliseconds per frame (multiple of 20 ms)
  % v_features  - Selected Features: Signal, FFT, Spectrogram, MFCC, LPC-AR
  % ----------------------------------------------------------------------------
  % Output:
  % ----------------------------------------------------------------------------
  % m_features  - the mixtures, one per row in a matrix_type
  % ----------------------------------------------------------------------------
  
  ##############################################################################
  # Internal Parameters
  ##############################################################################
  
  s_mfcc_parameters   = 'E0';
  n_ar_coefficients   = 12;
  n_ar_frame_ms       = 25;
  n_fft_stop_freq     = 4000;
  n_env_resample_freq = fs/8;
  
  ##############################################################################
  # Compute Sizes and Allocate Memory
  ##############################################################################
  
  debug = 0;
  
  n_frame_size      = fs / 1000 * frame_ms;
  n_train_test_size = size(m_mixtures, 1);  
  n_feature_size    = 0;
  
  % Generate dummy signal to compute the number of features
  test_f = randn(1, n_frame_size);
  
  % 1 - Unprocessed Signal
  if (v_features(1) == 1)
    n_feature_size += n_frame_size;
  end
  
  % 2 - FFT
  if (v_features(2) == 1)
    test_FFT = get_speech_spectrum(test_f, fs, n_fft_stop_freq);
    n_feature_size += length(test_FFT);
  end
  
  % 3 - Spectrogram
  v_f = [];
  v_t = [];  
  if (v_features(3) == 1)
    [test_S, v_f, v_t] = get_speech_spectrogram(test_f, fs);
    n_feature_size += length(test_S);    
  end
  
  % 4 - MFCC
  if (v_features(4) == 1)
    test_M = melcepst(test_f, fs, s_mfcc_parameters)(:);    
    n_feature_size += length(test_M);
  end
  
  % 5 - AR Coefficients
  if (v_features(5) == 1)
    test_AR = lpcauto(test_f, n_ar_coefficients, n_ar_frame_ms * fs / 1000);
    test_AR = test_AR(:, 2 : end)(:);
    n_feature_size += length(test_AR);
  end
  
  % 6 - Signal Envelope, computed w/ Hilbert and decimated
  if (v_features(6) == 1)
    test_SE = get_speech_envelope(test_f, fs, n_env_resample_freq);
    n_feature_size += length(test_SE);
  end
  
  % 7 - Power Spectral Density
  if (v_features(7) == 1)
    [test_PSD, F] = periodogram(test_f, [], length(test_f), fs);
    n_feature_size += length(test_PSD);
  end
  
  % 8 - Histogram
  if (v_features(8) == 1)
    f_min = min(test_f);
    f_max = max(test_f);
    test_hist = get_histogram(test_f, f_min, f_max, 50);
    n_feature_size += length(test_hist);
  end
  
  % Allocate memory
  m_features  = zeros(n_train_test_size, n_feature_size);
  
  s0 = time();
  
  n_progress_step = 5000;
  
  for i = 1 : n_train_test_size
  
    ############################################################################
    # Produce Selected Features
    ############################################################################
    
    % MFCC has some bugs and produces infinite values in some cases
    b_finite = 1;
    
    v_feature = [];
    v_mixed   = m_mixtures(i, :);
    
    if (v_features(1) == 1)
      v_feature = [v_feature, v_mixed];
    end
        
    if (v_features(2) == 1)
      v_fft = get_speech_spectrum(v_mixed, fs, n_fft_stop_freq);
      v_feature = [v_feature, v_fft];
    end
    
    if (v_features(3) == 1)
      v_spectrogram = get_speech_spectrogram(v_mixed, fs);
      v_feature = [v_feature, v_spectrogram'];
    end
    
    if (v_features(4) == 1)
      v_mfcc = melcepst(v_mixed, fs, s_mfcc_parameters)(:);
      
      if (sum(isfinite(v_mfcc)) ~= length(v_mfcc))
        b_finite = 0;
        printf("Mixture %d resulted in Inf.\n", i);
        fflush(stdout);
     
        continue;   
      end
      
      v_feature = [v_feature, v_mfcc'];        
    end
    
    if (v_features(5) == 1)
      v_ar = lpcauto(v_mixed, n_ar_coefficients, n_ar_frame_ms * fs / 1000);
      v_ar = v_ar(:, 2 : end)(:);
      v_feature = [v_feature, v_ar'];
    end
    
    if (v_features(6) == 1)
      v_env = get_speech_envelope(v_mixed, fs, n_env_resample_freq);
      v_feature = [v_feature, v_env];
    end
    
    if (v_features(7) == 1)
      [v_psd, F] = periodogram(v_mixed, [], length(v_mixed), fs);
      v_feature = [v_feature, v_psd'];
    end
    
    if (v_features(8) == 1)
      f_min = min(v_mixed);
      f_max = max(v_mixed);
      v_hist = get_histogram(v_mixed, f_min, f_max, 50);
      v_feature = [v_feature, v_hist];
    end      
    
    % Save feature
    if (b_finite == 1)
      m_features(i, :) = v_feature;
    end
    
    if (mod(i, n_progress_step) == 0)
      printf("Processing %d mixture.\n", i);
      fflush(stdout);
    end
    
  end
  
  s1 = time();
  printf("Feature creation: %.3f sec.\n", s1 - s0);
  fflush(stdout);
  
  ##############################################################################
  # Debug Plots
  ##############################################################################
  
  if (debug == 1)  
    figure();
    for i = 1 : 6                
      n_test = randi(n_train_test_size);      
      subplot(3, 2, i);
      plot(m_features(n_test, :)); grid;     
    end  
  end  

endfunction