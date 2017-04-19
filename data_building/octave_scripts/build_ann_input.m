% Build features

addpath("/home/valentin/Working/phd_project/build_dataset/scripts");
load("filenames.mat");
pkg load signal;

% Select parameters
n_max_files         = 2420;
v_files             = all_files(1 : n_max_files);
fs                  = 16000;
frame_ms            = 100;
frame_inc_ms        = 50;
frame_ms_vad	    = 500;
v_features          = [0, 0, 0, 1, 1, 0];
b_add_square_feats  = 1;
f_speech_threshold  = 0.95;
s_path              = "/home/valentin/Working/saa_experiments_db/naa_corpus/audio_micSamson_parts/";
b_just_sizes        = 0;

% Build features
[m_features, v_files_link, v_files_frames] = build_features_multiple_files(v_files, ...
  frame_ms, frame_inc_ms, frame_ms_vad, fs, v_features, f_speech_threshold, s_path, b_just_sizes);

% Save current files
save("-ascii", "files_frames.txt", "v_files_frames");

if (b_just_sizes == 0)
  % Add second degree polynomial features
  if (b_add_square_feats == 1)
    n_features = size(m_features)(2);
    n_samples = size(m_features)(1);

    for i = 1 : n_features
      v_square_feat = m_features(:, i) .^ 2;
      m_features = [m_features, v_square_feat];
    end
    
  end

  % Normalize features to the same values we used for training set
  [m_feat_normalized] = do_feature_scaling(m_features, "mmm_train.txt");


  % Save features file
  save("-ascii", "files_link.txt", "v_files_link");
  save("-ascii", "features.txt", "m_feat_normalized");
end
