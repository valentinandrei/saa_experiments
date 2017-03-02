# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_features] = build_features (m_mixtures, fs, frame_ms, v_features)

  addpath ("/home/valentin/Working/sw_tools/voicebox");

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
  n_ar_frame_ms       = 15;
  n_fft_stop_freq     = 4000;
  n_env_resample_freq = fs/8;
  
  ##############################################################################
  # Compute Sizes and Allocate Memory
  ##############################################################################
  
  debug = 1;
  
  n_frame_size      = fs / 1000 * frame_ms;
  n_train_test_size = size(m_mixtures, 1);  
  n_feature_size    = 0;
  
  % - Unprocessed Signal
  if (v_features(1) == 1)
    n_feature_size += n_frame_size;
  end
  
  % 1 - FFT
  if (v_features(2) == 1)
    test_f = randn(1, n_frame_size);
    test_FFT = get_speech_spectrum(test_f, fs, n_fft_stop_freq);
    n_feature_size += length(test_FFT);
  end
  
  % 2 - Spectrogram
  v_f = [];
  v_t = [];  
  if (v_features(3) == 1)
    test_f = randn(1, n_frame_size);
    [test_S, v_f, v_t] = get_speech_spectrogram(test_f, fs);
    n_feature_size += length(test_S);    
  end
  
  % 3 - MFCC
  if (v_features(4) == 1)
    test_f = randn(1, n_frame_size);
    test_M = melcepst(test_f, fs, s_mfcc_parameters)(:);    
    n_feature_size += length(test_M);
  end
  
  % 4 - AR Coefficients
  if (v_features(5) == 1)
    test_f = randn(1, n_frame_size);
    test_AR = lpcauto(test_f, n_ar_coefficients, n_ar_frame_ms * fs / 1000);
    test_AR = test_AR(:, 2 : end)(:);
    n_feature_size += length(test_AR);
  end
  
  % 5 - Signal Envelope, computed w/ Hilbert and decimated
  if (v_features(6) == 1)
    test_f = randn(1, n_frame_size);
    test_SE = get_speech_envelope(test_f, fs, n_env_resample_freq);
    n_feature_size += length(test_SE);
  end
  
  % Allocate memory
  m_features  = zeros(n_train_test_size, n_feature_size);
  
  s0 = time();
  
  for i = 1 : n_train_test_size
    ############################################################################
    # Produce Selected Features
    ############################################################################
    
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
      end
      v_feature = [v_feature, v_mfcc'];        
    end
    
    if (v_features(5) == 1)
      v_ar = lpcauto(v_mixed, n_ar_coefficients, n_ar_frame_ms * fs / 1000);
      v_ar = v_ar(:, 2 : end)(:);
      v_feature = [v_feature, v_ar'];
    end
    
    % 5 - Signal Envelope, computed w/ Hilbert and decimated
    if (v_features(6) == 1)
      v_env = get_speech_envelope(v_mixed, fs, n_env_resample_freq);
      v_feature = [v_feature, v_env];
    end
    
    % Save feature
    if (b_finite == 1)
      m_features(i, :) = v_feature;
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