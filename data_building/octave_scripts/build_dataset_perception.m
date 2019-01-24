# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

addpath("E:/1_Proiecte_Curente/1_Speaker_Counting/3rdparty/voicebox");
pkg load signal
pkg load ltfat

% ------------------------------------------------------------------------------

v_dir_database  = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_test_clean/test-clean-trimmed/*';
n_max_speaker_directories = 50;

% ------------------------------------------------------------------------------

fs                  = 16000;
v_frame_ms          = [5000, 2000, 1000, 500];
frame_inc_ms        = 250;
n_classes           = 4;
n_max_speakers      = 4;
n_samples_pc_train  = 5;
n_samples_pc_test   = 5;
with_reverb         = 0;

% ------------------------------------------------------------------------------

v_directories = glob(strcat(v_dir_database));

% Create Speech Mixtures

n_train_size = n_max_speakers * n_samples_pc_train;
n_test_size = n_max_speakers * n_samples_pc_test;
n_total_size = n_train_size + n_test_size;

% Preallocate cell

cMixturesNoReplay = cell(length(v_frame_ms), 2);
cMixturesReplay = cell(length(v_frame_ms), 2);

% For each framelength

for i = 1 : length(v_frame_ms)
  
  % Size of the signal window
  n_frame_size  = fs/1000 * v_frame_ms(i);
  
  % Process Speech Inputs
  [c_speech, v_n_frames_speech, n_speakers_recorded] = build_speech_input ...
    ( v_directories, fs, v_frame_ms(i), frame_inc_ms, n_max_speaker_directories);
   
  % Count eligible speakers
  n_eligible_speakers = sum(v_n_frames_speech ~= 0);
  if (n_eligible_speakers < n_max_speakers)
    printf("Not enough speakers with speech frames sufficiently long.\n");
    fflush(stdout);    
    return;
  else
    printf("Eligible speakers = %d\n", n_eligible_speakers);
    fflush(stdout);    
  end
  
  % Select only speakers for which we could extract frames sufficiently long
  c_speech_eligible = cell(n_eligible_speakers, 1);
  v_n_frames_eligible = zeros(n_eligible_speakers, 1);
  k = 1;
  for j = 1 : length(v_n_frames_speech)
    if (v_n_frames_speech(j) ~= 0)
      c_speech_eligible{k} = c_speech{j};
      v_n_frames_eligible(k) = v_n_frames_speech(j);
      k++;
    end
  end

  % Create Train Mixtures
  [m_mixtures, v_labels] = build_mixtures ...
    ( c_speech_eligible, v_n_frames_eligible, ...
      n_total_size, n_max_speakers, n_eligible_speakers, ...
      n_frame_size, with_reverb, 1);     

  % Save Mixtures to Cell
  cMixturesNoReplay{i, 1} = m_mixtures(1 : n_train_size, :);
  cMixturesNoReplay{i, 2} = v_labels(1 : n_train_size, :);
  cMixturesReplay{i, 1} = m_mixtures(n_train_size + 1 : end, :);
  cMixturesReplay{i, 2} = v_labels(n_train_size + 1 : end, :);
  
end

% Save cell arrays

save mixtures_no_replay.mat cMixturesNoReplay
save mixtures_replay.mat cMixturesReplay