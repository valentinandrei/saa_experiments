# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_mixtures, v_labels] = build_mixtures ...
  ( c_speech, v_n_frames_speech, ...
    n_set_size, n_max_speakers, n_files, ...
    n_frame_size, with_reverb, count_speakers)

  % ----------------------------------------------------------------------------     
  % Input
  % ----------------------------------------------------------------------------
  % c_speech          - Cell array with each speech frame per speaker
  % v_n_frames_speech - Frame count per speaker
  % n_set_size        - The number of mixtures to be created
  % n_max_speakers    - Maximum number of speakers in a mix
  % n_files           - Number of single speaker filesep
  % n_frame_size      - The length of the frame in #samples
  % with_reverb       - Enable/Disable reverberation effects
  % count_speakers    - If 1, the label is an array where v(k) is 1 if k 
  %                     speakers are active 
  % ----------------------------------------------------------------------------
  % Output
  % ----------------------------------------------------------------------------
  % m_mixtures        - Matrix with mixtures, one per row
  % v_labels          - Labels vector / matrix_type
  % ----------------------------------------------------------------------------

  t0 = time();
  
  n_classes = 1;
  if (count_speakers == 1)
    n_classes = n_max_speakers;
  end
  
  m_mixtures = zeros(n_set_size, n_frame_size);  
  v_labels   = zeros(n_set_size, n_classes); 
  n_speakers = 1;
  n_label    = 0;
  
  for i = 1 : n_set_size
    
    % Select number of speakers
    n_speakers = 1;
    if (count_speakers == 1)
      n_speakers = randi(n_max_speakers) + 1;
      v_labels(i, n_speakers) = 1.0;
    else
      single_multi = randi(2);
       
      if (single_multi ~= 1)
        n_speakers = 1 + randi(n_max_speakers - 1);
        v_labels(i) = 1.0;
      end
    end 

    % Get n_speakers different indexes    
    v_speakers = do_n_diff_rand(n_speakers, 1, n_files);
    
    % Collect all single speech frames in a matrix
    m_single = zeros(n_speakers, n_frame_size);
    for j = 1 : n_speakers
      idx_frame = randi(v_n_frames_speech(v_speakers(j)), 1);
      m_single(j, :) = c_speech{v_speakers(j)}(idx_frame, :);
    end

    % Mix all signals in the matrix      
    if (with_reverb == 0)
      m_mixtures(i, :) = do_mix_non_reverb(m_single);
    else
      % TODO
    end
  end
  
  t1 = time();
  printf("build mixtures duration: %.3f seconds\n", t1 - t0);
  fflush(stdout);

endfunction