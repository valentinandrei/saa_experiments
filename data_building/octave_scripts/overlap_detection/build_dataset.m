# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

addpath ("/home/valentin/Working/phd_project/build_dataset/wavfiles");
pkg load signal

wavfiles        = { "/home/valentin/Working/phd_project/build_dataset/wavfiles/S1.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S2.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S3.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S4.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S5.wav", ... 
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S6.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S7.wav", ...                     
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S8.wav"};                    

#{                    
wavfiles        = { "/home/valentin/Working/phd_project/build_dataset/wavfiles/S9.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S10.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S13.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S15.wav"};                   
#}
#{
wavfiles        = { "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav"};                                                                                                         
#}

fs                  = 16000;
frame_ms            = 25;
frame_inc_ms        = 5;
n_max_speakers      = 3;
n_samples_per_count = 25000;
with_reverb         = 0;
b_add_square_feats  = 1;
b_normalize         = 1;
b_do_pca_analysis   = 0;

% Specify selected features:
%   Entire Signal
%   FFT
%   Spectrogram
%   MFCC ('E0')
%   AR_Coefficients (12 coefs for each 15 ms window)
%   Decimated Speech Signal Envelope

v_features          = [0, 1, 0, 1, 1, 1];

# Collect Training Features

[m_features, v_labels] = build_features ( wavfiles, n_max_speakers, ...
  n_samples_per_count, fs, frame_ms, frame_inc_ms, with_reverb, v_features);
  
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

# Training Features' Mean Normalization and Scaling

if (b_normalize == 1)
  v_max       = max (m_features);
  v_min       = min (m_features);
  v_mean      = mean(m_features);
  m_features  = (m_features - v_mean) ./ (v_max - v_min);
  m_mmm       = [v_max; v_min; v_mean];
  save("-ascii", "mmm_train.txt", "m_mmm");
  save("-ascii", "x_train_normalized.txt", "m_features");
else
  save("-ascii", "x_test_unnormalized.txt", "m_features");
end

save("-ascii", "y_train.txt", "v_labels");