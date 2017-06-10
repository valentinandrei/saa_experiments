# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

addpath ("/home/valentin/Working/saa_experiments_db/valentin_recordings/");
pkg load signal

% ------------------------------------------------------------------------------

wavfiles  = { "/home/valentin/Working/saa_experiments_db/valentin_recordings/S1.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S2.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S3.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S4.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S5.wav", ... 
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S6.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S7.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S8.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S9.wav"};

#{
wavfiles  = { "/home/valentin/Working/saa_experiments_db/valentin_recordings/S10.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S11.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S13.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/S14.wav"};
#}

#{
wavfiles  = { "/home/valentin/Working/saa_experiments_db/valentin_recordings/TEST.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/TEST.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/TEST.wav", ...
              "/home/valentin/Working/saa_experiments_db/valentin_recordings/TEST.wav"};
#}

% ------------------------------------------------------------------------------

fs                  = 16000;
frame_ms            = 100;
frame_inc_ms        = 50;
n_classes           = 5;
n_max_speakers      = 5;
n_samples_per_count = 20000;
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

v_features  = [0, 1, 0, 1, 1, 1, 0];

% ------------------------------------------------------------------------------

% Process Speech Inputs
[c_speech, v_n_frames_speech] = build_speech_input ...
  ( wavfiles, fs, frame_ms, frame_inc_ms);

% Create Speech Mixtures
n_set_size = (n_classes + 1) * n_samples_per_count;
if (count_speakers == 1)
  n_set_size = n_max_speakers * n_samples_per_count;
end
n_files       = length(wavfiles);
n_frame_size  = fs/1000 * frame_ms;

[m_mixtures, v_labels] = build_mixtures ...
  ( c_speech, v_n_frames_speech, ...
    n_set_size, n_max_speakers, n_files, ...
    n_frame_size, with_reverb, count_speakers);

% Create Features from Mixtures
m_features = build_features (m_mixtures, fs, frame_ms, v_features);

% ------------------------------------------------------------------------------

% Add second degree polynomial features
if (b_add_square_feats == 1)

  n_features = size(m_features)(2);
  n_samples = size(m_features)(1);

  for i = 1 : n_features
    v_square_feat = m_features(:, i) .^ 2;
    m_features = [m_features, v_square_feat];
  end
  
end

# Principal Component Analysis
if (b_do_pca_analysis)

end

% ------------------------------------------------------------------------------

% Features' Mean Normalization and Scaling
if (b_train == 1)

  [m_features_norm, mu, sigma] = do_feature_normalization(m_features);
  save("-ascii", "x_train_normalized.txt", "m_features_norm");
  save("-ascii", "y_train.txt", "v_labels");
  
else

  [m_features_norm, mu, sigma] = do_feature_normalization(m_features);
  save("-ascii", "x_test_normalized.txt", "m_features");
  save("-ascii", "y_test.txt", "v_labels");  
  
end