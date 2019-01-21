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
  % n_files           - Number of single speaker files
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
  
  m_mixtures    = zeros(n_set_size, n_frame_size);  
  v_labels      = zeros(n_set_size, n_classes);
  n_concurrent  = 1;
  n_label       = 0;
  n_errors      = 0;
  
  i = 1;
  while (i <= n_set_size)

    % Select number of speakers
    n_concurrent = 1;
    if (count_speakers == 1)
      n_concurrent = randi(n_max_speakers);
      v_labels(i, n_concurrent) = 1.0;
    else
      single_multi = randi(2);
       
      if (single_multi ~= 1)
        n_concurrent = randi(n_max_speakers);
        v_labels(i) = 1.0;
      end
    end

    % Get n_concurrent different indexes    
    v_speakers = do_n_diff_rand(n_concurrent, 1, n_files);
    if (length(v_speakers) == 0)
      error("Issues when generating random speaker pairs.");
    endif
    
    % Collect all single speech frames in a matrix
    m_single = zeros(n_concurrent, n_frame_size);
    for j = 1 : n_concurrent
      idx_frame = randi(v_n_frames_speech(v_speakers(j)));
      m_single(j, :) = c_speech{v_speakers(j)}(idx_frame, :);
    end

    % Mix all signals in the matrix      
    if (with_reverb == 0)
      if (sum(m_single(:)) ~=0)
        v_mixture = do_mix_non_reverb(m_single, 1);
        m_mixtures(i, :) = v_mixture;
        i = i + 1;       
      else
        n_errors = n_errors + 1;
      endif
    else
      % TODO
    end    
  endwhile
  
  t1 = time();
  printf("build mixtures duration: %.3f seconds\n", t1 - t0);
  printf("number of errors (Inf values): %d\n", n_errors);
  fflush(stdout);

endfunction