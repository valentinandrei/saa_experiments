# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_features, v_labels] = build_features(v_wavfiles, ...
  n_max_speakers, n_samples_per_count, fs, frame_ms, frame_inc_ms, ...
  with_reverb, v_features)

  % Usage:
  %
  % This function takes as input a number of wavfiles and creates a set of
  % mixture frames labeled with the number of competing speakers.
  %
  % Input:
  %
  % m_wavfiles          - the array with the name of all the wavfiles
  % n_max_speakers      - the maximum number of speakers per mixture
  % n_samples_per_count - the number of mixtures for each targeted count
  % fs                  - targeted sampling frequency in Hz
  % frame_ms            - the number of milliseconds per frame (multiple of 20 ms)
  % frame_inc_ms        - the increment per frame in milliseconds
  % with_reverb         - if 1, enables reverberation inclusion in mixing
  % v_features          - Selected Features: Signal, FFT, Spectrogram, MFCC, LPC-AR
  %
  % Output:
  %
  % m_features          - the mixtures, one per row in a matrix_type
  % v_labels            - row vector with corresponding speaker count per mixture
  
  debug = 1;
  
  n_files = length(v_wavfiles);
  n_frame_size = fs / 1000 * frame_ms;
  
  % Size of the dataset. Silence is also accounted for
  
  n_train_test_size = n_samples_per_count * 2;
  
  % Read single speaker filesep
  
  c_speech = {};
  v_n_frames_speech = zeros(n_files, 1);
  
  ##############################################################################
  # Sepparate Silence and Speech
  ##############################################################################
  
  for i = 1 : n_files
  
    [s, start, stop, act] = build_vad_mask(v_wavfiles{i}, ...
      fs, frame_ms, frame_inc_ms);
      
    [m_speech, n_speech, m_silence, n_silence] = ...
      get_speech_silence_frames(s, start, stop, act);
      
    c_speech{i} = m_speech;
    v_n_frames_speech(i) = n_speech;
      
  end
  
  ##############################################################################
  # Compute Sizes and Allocate Memory
  ##############################################################################
  
  n_feature_size = 0;
  
  % - Unprocessed Signal
  if (v_features(1) == 1)
    n_feature_size += n_frame_size;
  end
  
  % 1 - FFT
  if (v_features(2) == 1)
    test_f = randn(1, n_frame_size);
    test_FFT = get_speech_spectrum(test_f, fs);
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
    test_M = melcepst(test_f, fs, 'E0')(:);    
    n_feature_size += length(test_M);
  end
  
  % 4 - AR Coefficients
  if (v_features(5) == 1)
    test_f = randn(1, n_frame_size);
    test_AR = lpcauto(test_f, 12, 15 * fs / 1000);
    test_AR = test_AR(:, 2 : end)(:);
    n_feature_size += length(test_AR);
  end
  
  % 5 - Signal Envelope, computed w/ Hilbert and decimated
  if (v_features(6) == 1)
    test_f = randn(1, n_frame_size);
    test_SE = get_speech_envelope(test_f, fs);
    n_feature_size += length(test_SE);
  end

  n_classes   = 1;
  m_features  = zeros(n_train_test_size, n_feature_size);
  v_mixed     = zeros(1, n_frame_size);
  v_labels    = zeros(n_train_test_size, n_classes);
  
  for i = 1 : n_train_test_size

    single_multi = randi(2);
    if (single_multi == 1)
      n_speakers = 1;      
    else
      n_speakers = 1 + randi(n_max_speakers - 1);
      v_labels(i) = 1.0;  % Signal multi speaker class
    end
    
    ############################################################################
    # Produce Speech Mixtures
    ############################################################################
          
    v_speakers = get_n_diff_rand(n_speakers, 1, n_files);
    v_frames = zeros(n_speakers, 1);
    for j = 1 : n_speakers
      v_frames(j) = randi(v_n_frames_speech(v_speakers(j)), 1);
    end
      
    m_single = zeros(n_speakers, n_frame_size);
    for j = 1 : n_speakers
      m_single(j, :) = c_speech{v_speakers(j)}(v_frames(j), :);
    end
      
    if (with_reverb == 0)
      v_mixed = mix_non_reverb(m_single);
    else
      % TODO
    end
       
    ############################################################################
    # Produce Selected Features
    ############################################################################
    
    v_feat_unproc_signal=[];
    v_feat_fft = [];
    v_feat_spectrogram = [];
    v_feat_mfcc = [];
    v_feat_ar = [];
    v_feat_env = [];
    
    if (v_features(1) == 1)
      v_feat_unproc_signal = [v_feat_unproc_signal, v_mixed];
    end
        
    if (v_features(2) == 1)
      v_fft = get_speech_spectrum(v_mixed, fs);
      v_feat_fft = [v_feat_fft, v_fft];
    end
    
    if (v_features(3) == 1)
      v_spectrogram = get_speech_spectrogram(v_mixed, fs);
      v_feat_spectrogram = [v_feat_spectrogram, v_spectrogram];
    end
    
    if (v_features(4) == 1)
      v_mfcc = melcepst(v_mixed, fs, 'E0')(:);
      v_feat_mfcc = [v_feat_mfcc, v_mfcc];
    end
    
    if (v_features(5) == 1)
      v_ar = lpcauto(v_mixed, 12, 15 * fs / 1000);
      v_ar = v_ar(:, 2 : end)(:);
      v_feat_ar = [v_feat_ar, v_ar];
    end
    
    % 5 - Signal Envelope, computed w/ Hilbert and decimated
    if (v_features(6) == 1)
      v_env = get_speech_envelope(v_mixed, fs);
      v_feat_env = [v_feat_env, v_env];
    end
    
    ############################################################################
    # Combine Features
    ############################################################################   
        
    m_features(i, :) = [v_feat_unproc_signal, v_feat_fft, v_feat_spectrogram', ...
      v_feat_mfcc', v_feat_ar', v_feat_env];
    
  end
  
  ##############################################################################
  # Debug Plots
  ##############################################################################
  
  if (debug == 1)  
    figure();
    for i = 1 : 6                
      n_test = randi(n_train_test_size);      
      subplot(3, 2, i);
      plot(m_features(n_test, :)); grid;
      xlabel(v_labels(n_test));      
    end  
  end

endfunction