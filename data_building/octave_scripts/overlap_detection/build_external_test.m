# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_features, m_labels] = build_external_test ...
  ( s_mixed, m_singles, fs, frame_ms, frame_inc_ms, b_use_mixer,
    v_features, b_count_speakers, b_add_square_feats, b_feature_scaling,
    s_file_scaling)
    
  % ----------------------------------------------------------------------------    
  % Input
  % ----------------------------------------------------------------------------
  % s_mixed             - Mixed speech signal
  % m_singles           - Matrix with single speech signals
  % fs                  - Sampling Frequency
  % frame_ms            - Frame milliseconds
  % frame_inc_ms        - Frame increment milliseconds
  % v_features          - Selected feature set
  % b_use_mixer         - If set to 1, call the mixer function on input signal
  % b_count_speakers    - If 1, count speakers per frame
  % b_add_square_feats  - If 1, add square features
  % b_feature_scaling   - If 1, do feature scaling
  % s_file_scaling      - Filename containing min, max, mean for scaling
  % ----------------------------------------------------------------------------
  % Output
  % ----------------------------------------------------------------------------
  % m_features          - The final feature matrix_type
  % m_labels            - Feature labels
  % ----------------------------------------------------------------------------  

n_max_speakers  = size(m_singles)(1);
n_samples       = size(m_singles)(2);
n_frame_samples = fs / 1000 * frame_ms;
n_inc_samples   = fs / 1000 * frame_inc_ms;
n_max_frames    = n_samples / n_inc_samples;
n_classes       = 1;

if (b_count_speakers == 1)
  n_classes = n_max_speakers;
end

if (n_samples ~= length(s_mixed))
  printf("Mixed signal not of the same size as single speech signals.");
  exit;
end

% Run the voice activity detection
m_vad_decisions = zeros(n_max_speakers, n_samples);
for i = 1 : n_max_speakers
  m_vad_decisions(i, :) = vadsohn(m_singles(i, :), fs);
end
v_final_vad = sum(m_vad_decisions);

% Pre-allocate with a larger size
m_mixed     = zeros(n_max_frames, n_frame_samples);
m_labels    = zeros(n_max_frames, n_classes);
n_index     = 1;
n_selected  = 0;

% Select mixtures
while (n_index + n_inc_samples < n_samples)

  i_start = n_index;
  i_stop  = i_start + n_frame_samples;
  f_vad   = sum(v_final_vad(i_start : i_stop)) / n_frame_samples;
  
  % Select sample
  for i = 1 : n_max_speakers
    if (f_vad == i)
      n_selected += 1;
      s_frame = s_mixed(i_start : i_stop);
      if (b_use_mixer)
        s_frame = do_mix_non_reverb(s_frame);
      end  
      m_mixed(n_selected, :) = s_frame;
      if (b_count_speakers == 1)
        m_labels(n_selected, i) = 1;
      else
        m_labels(n_selected) = 1;
      end
      break;
    end
  end
  
  n_index += n_inc_samples;  

end

% Trim containers
m_mixed     = m_mixed(1 : n_selected, :);
m_labels    = m_labels(1 : n_selected, :);

% Build Features
m_features  = build_features(m_mixed, fs, frame_ms, v_features);

% Polynomial Features
if (b_add_square_feats == 1)
  n_features  = size(m_features)(2);
  for i = 1 : n_features
    v_square_feat = m_features(:, i) .^ 2;
    m_features = [m_features, v_square_feat];
  end
end

% Feature Scaling
if (b_feature_scaling == 1)
  m_features = do_feature_scaling(m_features, s_file_scaling);
end

endfunction