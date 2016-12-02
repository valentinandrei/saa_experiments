# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_features, v_labels] = build_labeled_features (v_wavfiles, ...
  n_max_speakers, n_samples_per_count, fs, frame_ms, frame_inc_ms, ...
  with_reverb, feature_type)

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
  % n_bits              - targeted number of bits per sample
  % frame_inc_ms        - the increment per frame in milliseconds
  % n_seconds           - the number of seconds to be analyzed
  % with_reverb         - if 1, enables reverberation inclusion in mixing
  % feature_type        - 0 - Signal, 1 - FFT, 2 - Spectrogram
  %
  % Output:
  %
  % m_features          - the mixtures, one per row in a matrix_type
  % v_labels            - row vector with corresponding speaker count per mixture
  
  debug = 1;
  
  n_files = length(v_wavfiles);
  n_frame_size = fs / 1000 * frame_ms;
  f_scale = 0.0;
  
  % Size of the dataset. Silence is also accounted for
  
  n_train_test_size = n_samples_per_count * (n_max_speakers + 1);
  
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
    
    f_scale_temp = max(s);
    if (f_scale < f_scale_temp)
      f_scale = f_scale_temp;
    end
      
  end
  
  ##############################################################################
  # Compute Sizes and Allocate Memory
  ##############################################################################
  
  n_feature_size = n_frame_size;
  if (feature_type == 1)
    n_feature_size = n_frame_size / 2;
  end
  
  if (feature_type == 2)
    test_f = randn(1, n_frame_size);
    test_S = get_speech_spectrogram(test_f, fs);
    n_feature_size = length(test_S);
  end
  
  n_classes = 1;
  if (feature_type == 2)
    n_classes = n_max_speakers + 1;
  end
  
  m_features = zeros(n_train_test_size, n_feature_size);
  v_mixed = zeros(1, n_frame_size);
  v_feature = zeros(1, n_feature_size);
  v_labels = zeros(n_train_test_size, n_classes);
  
  for i = 1 : n_train_test_size

    n_speakers = randi(n_max_speakers);
    
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
      v_mixed = mix_non_reverb(m_single, f_scale);
      v_feature = zeros(1, length(v_mixed));
    else
      % TODO
    end
       
    ############################################################################
    # Produce Selected Features
    ############################################################################
    
    if (feature_type == 0)
      v_feature = v_mixed;
      v_labels(i) = n_speakers;
    end
        
    if (feature_type == 1)
      v_fft_mixed = abs(fft(v_mixed));
      N = length(v_fft_mixed);
      v_feature = v_fft_mixed(1 : N/2);
      v_labels(i) = n_speakers;
    end
    
    if (feature_type == 2)
      v_feature = get_speech_spectrogram(v_mixed, fs);
      v_labels(i, n_speakers) = 1.0;
    end
        
    m_features(i, :) = v_feature;
    
  end
  
  ##############################################################################
  # Debug Plots
  ##############################################################################
  
  if (debug == 1)
  
    for i = 1 : 6
    
      n_test = randi(n_train_test_size);      
      subplot(3, 2, i);
      plot(m_features(n_test, :)); grid;      
      x_val = v_labels(n_test);
      
      if (feature_type == 2)
        [max_val, x_val] = max(v_labels(n_test, :));
        x_val = x_val - 1;
      end
      
      xlabel(x_val);
    end
  
  end

endfunction