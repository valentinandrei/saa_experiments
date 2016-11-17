# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

addpath ("/home/valentin/Working/phd_project/build_dataset/wavfiles");

wavfiles            = [];
fs                  = 16000;
frame_ms            = 100;
frame_inc_ms        = 40;
n_bits              = 16;
n_max_speakers      = 3;
n_samples_per_count = 10000;
n_seconds           = 150;
with_reverb         = 0;

[m_mixtures, v_labels] = build_labeled_mixtures (wavfiles, ...
  n_max_speakers, n_samples_per_count, n_seconds, ...
  fs, frame_ms, frame_inc_ms, n_bits, with_reverb);
  
save("-ascii", "train_mixtures.txt", "m_mixtures");
save("-ascii", "train_labels.txt", "v_labels");
