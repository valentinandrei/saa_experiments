# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

addpath ("/home/valentin/Working/phd_project/build_dataset/wavfiles");
pkg load signal

wavfiles_train  = { "/home/valentin/Working/phd_project/build_dataset/wavfiles/S1.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S2.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S3.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S4.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S5.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S6.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S7.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S8.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S9.wav"};
                       
wavfiles_test   = { "/home/valentin/Working/phd_project/build_dataset/wavfiles/S10.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S11.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S12.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S13.wav", ...
                    "/home/valentin/Working/phd_project/build_dataset/wavfiles/S14.wav", ...
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
n_samples_per_count = 25000;
with_reverb         = 0;
feature_type        = 3;

# Collect Training Features

[m_features_train, v_labels_train] = build_features ( wavfiles_train, n_max_speakers, ...
  n_samples_per_count, fs, frame_ms, frame_inc_ms, with_reverb, feature_type);
  
# Training Features' Normalization
  
v_max             = max (m_features_train);
v_min             = min (m_features_train);
v_mean            = mean(m_features_train);
m_features_train  = (m_features_train - v_mean) ./ (v_max - v_min);
  
save("-ascii", "x_mfcc_S1_9_500ms_16kHz.txt", "m_features_train");
save("-ascii", "y_mfcc_S1_9_500ms_16kHz.txt", "v_labels_train");

# Collect Inference Features

n_samples_per_count = 10000;

[m_features_test, v_labels_test] = build_features ( wavfiles_test, n_max_speakers, ...
  n_samples_per_count, fs, frame_ms, frame_inc_ms, with_reverb, feature_type);
  
# Training Inference' Normalization
  
v_max             = max (m_features_test);
v_min             = min (m_features_test);
v_mean            = mean(m_features_test);
m_features_test   = (m_features_test - v_mean) ./ (v_max - v_min);
  
save("-ascii", "x_mfcc_S10_15_500ms_16kHz.txt", "m_features_test");
save("-ascii", "y_mfcc_S10_15_500ms_16kHz.txt", "v_labels_test");
