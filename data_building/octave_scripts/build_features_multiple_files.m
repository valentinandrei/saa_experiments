# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_features, v_files_link, v_files_frames] = build_features_multiple_files (c_wavfiles, ...
  frame_ms, frame_inc_ms, fs, v_features, speech_threshold, s_path, b_just_sizes)
  
  addpath ("/home/valentin/Working/sw_tools/voicebox");  

  n_files = length(c_wavfiles);
  n_frame = floor(fs * frame_ms / 1000);
  n_frame_inc = floor(fs * frame_inc_ms / 1000);
  
  % Preallocation trick so that the code runs faster
  n_estimated_mixtures = n_files * 4;
  m_mixtures = zeros(n_estimated_mixtures, n_frame);
  v_files_link = zeros(n_estimated_mixtures, 1);
  v_files_frames = zeros(n_files, 1);
  n_mixtures = 1;
  
  for i = 1 : n_files
  
    s_filename = c_wavfiles{i};
    s_full_path = strcat(s_path, s_filename);
    
    [sig_wav, fs_orig, N] = wavread(s_full_path);
    if (fs_orig != fs)
      sig_speech = resample(sig_wav(:, 1), fs, fs_orig);
    end
    
    n_sample_length = length(sig_speech);
    n_sig_idx = 1;
    
    n_frames = floor(n_sample_length / n_frame_inc) - 1;
    v_files_frames(i) = n_frames;
    
    if (b_just_sizes == 0)
      while ((n_sig_idx + n_frame - 1) <= n_sample_length)
      
        % Verify if the selected frame contains voice activity
        
        v_mixture = sig_speech(n_sig_idx : n_sig_idx + n_frame - 1);
        vad_samples = sum(vadsohn(v_mixture, fs));
        b_speech = 0;
        if ((vad_samples / length(v_mixture)) > speech_threshold)
          b_speech = 1;
        end
        
        % Save only speech samples as silence has totally different numerical properties
        
        if (b_speech == 1)
          if (n_mixtures < n_estimated_mixtures)
            m_mixtures(n_mixtures, :) = v_mixture;
            v_files_link(n_mixtures) = i;
          else
            
            % Allocate more space
            m_new_space = zeros(n_files * 4, n_frame);
            v_new_space = zeros(n_files * 4, 1);
            
            n_estimated_mixtures = n_estimated_mixtures + (n_files * 4);
            m_mixtures = [m_mixtures; m_new_space];
            v_files_link = [v_files_link; v_new_space];
            
            m_mixtures(n_mixtures, :) = v_mixture;
            v_files_link(n_mixtures) = i;
          end

          n_mixtures = n_mixtures + 1;
        end
        
        % Advance to the next frame
        
        n_sig_idx = n_sig_idx + n_frame_inc;
      
      end
    end
    
    printf("Finished processing file %i ...\n", i);
    fflush(stdout);
  
  end
  
  if (b_just_sizes == 0)
    % Save only the corresponding mixtures
    m_mixtures = m_mixtures(1 : n_mixtures, :);
    v_files_link = v_files_link(1 : n_mixtures, :);

    % Compute features
    m_features = build_features (m_mixtures, fs, frame_ms, v_features);  
  end
  
endfunction
