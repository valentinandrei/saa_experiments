# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

addpath ("/home/valentin/Working/phd_project/build_dataset/wavfiles");
pkg load signal
                    
wavfiles_train  = { "/home/valentin/Working/phd_project/build_dataset/wavfiles/S10.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S11.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S13.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S15.wav"};                    
                        
                    #{
wavfiles_debug  = { "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav"};                                                                   
                    #}
                        
fs                  = 16000;
frame_ms            = 500;
frame_inc_ms        = 250;
n_max_speakers      = 3;
n_samples_per_count = 10000;
with_reverb         = 0;
feature_type        = 3;
b_add_square_feats  = 0;
b_normalize         = 0;

# Collect Training Features

[m_features_train, v_labels_train] = build_features ( wavfiles_train, n_max_speakers, ...
  n_samples_per_count, fs, frame_ms, frame_inc_ms, with_reverb, feature_type);
  
if (b_add_square_feats == 1)
  n_features = size(m_features_train)(2);
  n_samples = size(m_features_train)(1);

  for i = 1 : n_features
    v_square_feat = m_features_train(:, i) .^ 2;
    m_features_train = [m_features_train, v_square_feat];
  end
  
end
  
# Training Features' Normalization

if (b_normalize == 1)
  v_max             = max (m_features_train);
  v_min             = min (m_features_train);
  v_mean            = mean(m_features_train);
  m_features_train  = (m_features_train - v_mean) ./ (v_max - v_min);
  m_mmm             = [v_max; v_min; v_mean];
  save("-ascii", "mmm_mfcc_sq_S1_9_500ms_16kHz.txt", "m_mmm");
end

save("-ascii", "x_mfcc_sq_S10_15_500ms_16kHz.txt", "m_features_train");
save("-ascii", "y_mfcc_sq_S10_15_500ms_16kHz.txt", "v_labels_train");