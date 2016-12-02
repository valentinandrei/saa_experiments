# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

addpath ("/home/valentin/Working/phd_project/build_dataset/wavfiles");
pkg load signal

wavfiles           = { "/home/valentin/Working/phd_project/build_dataset/wavfiles/S1.wav", ...
                       "/home/valentin/Working/phd_project/build_dataset/wavfiles/S2.wav", ...
                       "/home/valentin/Working/phd_project/build_dataset/wavfiles/S3.wav", ...
                       "/home/valentin/Working/phd_project/build_dataset/wavfiles/S4.wav", ...
                       "/home/valentin/Working/phd_project/build_dataset/wavfiles/S5.wav", ...
                       "/home/valentin/Working/phd_project/build_dataset/wavfiles/S6.wav", ...
                       "/home/valentin/Working/phd_project/build_dataset/wavfiles/S7.wav", ...
                       "/home/valentin/Working/phd_project/build_dataset/wavfiles/S8.wav", ...
                       "/home/valentin/Working/phd_project/build_dataset/wavfiles/S9.wav"};
                        
                        #{
wavfiles            = { "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                        "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                        "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                        "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                        "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                        "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                        "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                        "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav", ...
                        "/home/valentin/Working/phd_project/build_dataset/wavfiles/TEST.wav"};                                             
                        #}
                        
fs                  = 16000;
frame_ms            = 200;
frame_inc_ms        = 100;
n_bits              = 16;
n_max_speakers      = 3;
n_samples_per_count = 5000;
with_reverb         = 0;
feature_type        = 2;

# Collect Features
[m_features, v_labels] = build_labeled_features (wavfiles, ...
  n_max_speakers, n_samples_per_count, fs, frame_ms, frame_inc_ms, ...
  with_reverb, feature_type);
  
save("-ascii", "x_spct_3_speakers_S1_9_200ms_100ms_inc_16kHz_5000.txt", "m_features");
save("-ascii", "y_spct_3_speakers_S1_9_200ms_100ms_inc_16kHz_5000.txt", "v_labels");
