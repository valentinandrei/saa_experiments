# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

addpath("E:/1_Proiecte_Curente/1_Speaker_Counting/3rdparty/voicebox");
pkg load signal
pkg load ltfat
% debug_on_warning(1);

% ------------------------------------------------------------------------------

v_dir_database  = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_test_clean/test-clean/*';
n_max_speaker_directories = 10;

% ------------------------------------------------------------------------------

fs                  = 16000;
frame_ms            = 1000;
frame_inc_ms        = 25;
n_classes           = 4;
n_max_speakers      = 4;
n_samples_per_count = 10;
n_set_size        = n_samples_per_count * n_max_speakers;
with_reverb         = 0;
count_speakers      = 1;
b_add_square_feats  = 0;
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

v_features  = [0, 0, 1, 0, 0, 1, 0, 1];

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

% Compute mixtures
[m_mixtures, v_labels_oh] = build_mixtures ...
  ( c_speech, v_n_frames_speech, ...
  n_set_size, n_max_speakers, n_speakers_recorded, ...
  n_frame_size, with_reverb, count_speakers);

% Transform one hot vector
[ix, iy] = find(v_labels_oh);
v_labels = iy;

% Run printing
b_example_available = 1;
while (b_example_available == 1)
  
  figure();
  
  n_speakers = 0;
  
  for i = 1 : n_max_speakers    
    for j = 1 : n_set_size      
      if (v_labels(j) == i)        
        subplot(2, 4, i);
        [S, v_f, v_t] = get_speech_spectrogram (m_mixtures(j, :), fs);
        imagesc(v_t, v_f, log(S));
        set (gca, "ydir", "normal");
        
        subplot(2, 4, i + 4);
        [s_envelope] = get_speech_envelope(m_mixtures(j, :), fs, fs / 128);
        plot(s_envelope); grid;
        
        n_speakers += 1
        v_labels(j) = 0;
 
        break; 
      end
    endfor
  endfor
  
  if (n_speakers == n_max_speakers)
    b_example_available = 1;
  else
    b_example_available = 0;
  end

endwhile

