# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

addpath("E:/1_Proiecte_Curente/1_Speaker_Counting/3rdparty/voicebox");
pkg load signal
pkg load ltfat
% debug_on_warning(1);

% ------------------------------------------------------------------------------

v_dir_database  = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/dev-clean/*';
n_max_speaker_directories = 5;

% ------------------------------------------------------------------------------

fs                  = 16000;
frame_ms            = 500;
frame_inc_ms        = 50;
n_classes           = 4;
n_max_speakers      = 4;
n_samples_per_count = 5000;
n_block_size        = 4000;
with_reverb         = 0;
count_speakers      = 1;
b_add_square_feats  = 0;
b_train             = 1;
b_do_pca_analysis   = 0;

% Specify selected features:
%   Entire Signal
%   FFT
%   Spectrogram
%   MFCC ('E0')
%   AR_Coefficients (12 coefs for each 15 ms window)
%   Decimated Speech Signal Envelope
%   Power Spectral Density
%   Histogram of the signal

v_features  = [0, 0, 1, 0, 0, 0, 0, 0];

% ------------------------------------------------------------------------------

v_directories = glob(strcat(v_dir_database));

% Process Speech Inputs
[c_speech, v_n_frames_speech, n_speakers_recorded] = build_speech_input ...
  ( v_directories, fs, frame_ms, frame_inc_ms, n_max_speaker_directories);

% Create Speech Mixtures
n_set_size = (n_classes + 1) * n_samples_per_count;
if (count_speakers == 1)
  n_set_size = n_max_speakers * n_samples_per_count;
end

% Size of the signal window
n_frame_size  = fs/1000 * frame_ms;

% Compute feature sets in blocks to avoid Out of Memory issues

n_current_set_size = 0;

while (n_current_set_size < n_set_size)
  
  [m_mixtures, v_labels] = build_mixtures ...
    ( c_speech, v_n_frames_speech, ...
      n_block_size, n_max_speakers, n_speakers_recorded, ...
      n_frame_size, with_reverb, count_speakers);      

  % Create Features from Mixtures
  m_features = build_features (m_mixtures, fs, frame_ms, v_features);

  % Here we just need to store the features
  clear m_mixtures

  % ----------------------------------------------------------------------------

  n_features = size(m_features)(2);
  n_samples = size(m_features)(1);

  % Add second degree polynomial features
  if (b_add_square_feats == 1)
    for i = 1 : n_features
      v_square_feat = m_features(:, i) .^ 2;
      m_features = [m_features, v_square_feat];
    end
  end

  # Principal Component Analysis
  if (b_do_pca_analysis)
    printf("Number of features before PCA: %d.", n_features);
    [m_features, sv, n_sv] = do_pca(m_features);
    n_features = size(m_features)(2);
    printf("Number of features after PCA: %d.", n_features);  
  end

  % ----------------------------------------------------------------------------
  
  % By default, we do feature normalization
  [m_features_norm, mu, sigma] = do_feature_normalization(m_features);
    
  % Here we don't need to store the unnormalized features anymore
  clear m_features  
  
  % Features' Mean Normalization and Scaling
  if (b_train == 1)
    save("-ascii", "-append", "x_train_normalized.txt", "m_features_norm");
    save("-ascii", "-append", "y_train.txt", "v_labels");
  else
    save("-ascii", "-append", "x_test_normalized.txt", "m_features");
    save("-ascii", "-append", "y_test.txt", "v_labels");      
  end
 
  % Free memory for processing the next block
  clear m_features_norm
  
  % Move to next block
  n_current_set_size += n_block_size
  
  % Show progress
  printf("Current set size = %d\n", n_current_set_size);
  fflush(stdout);
    
endwhile

% Here we don't need to store the single speech frames into memory
clear c_speech
clear v_n_frames_speech
