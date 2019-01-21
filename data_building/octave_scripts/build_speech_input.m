# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [c_speech, v_n_frames_speech, n_speakers] = build_speech_input ...
    ( v_directories, fs, frame_ms, frame_inc_ms, max_speakers = 0)

  % ----------------------------------------------------------------------------
  % Input:
  % ----------------------------------------------------------------------------
  % v_directories     - the directories containing audio files for each speaker
  % fs                - targeted sampling frequency in Hz
  % frame_ms          - the number of milliseconds per frame (multiple of 20 ms)
  % frame_inc_ms      - the increment per frame in milliseconds
  % ----------------------------------------------------------------------------
  % Output:
  % ----------------------------------------------------------------------------
  % c_speech          - cell array with all speaker speech frames per cell
  % v_n_frames_speech - speech frames count per speaker
  
  n_speakers        = length(v_directories);
  if (max_speakers ~= 0)
    if (n_speakers > max_speakers)
      n_speakers = max_speakers;
    endif
  endif
  
  c_speech          = {};
  v_n_frames_speech = zeros(n_speakers, 1);
  
  ##############################################################################
  # Sepparate Silence and Speech
  ##############################################################################
  
  s0 = time();
  
  % Loop across all speaker directories
  for i = 1 : n_speakers
    
    m_speaker_speech = [];
    n_speaker_samples = 0;
    
    printf("Folder: %s\n", v_directories{i});
    fflush(stdout);
    
    sessions_list = glob(strcat(v_directories{i}, "/*"));
    
    % Loop across all speaker sessions
    for j = 1 : length(sessions_list)
      
      file_list = glob(strcat(sessions_list{j}, "/*.flac"));
      
      % Loop across all speaker recordings
      for k = 1: length(file_list)
        
        s_file = file_list{k};
        
        [s, start, stop, act] = get_speech_vad_mask(s_file, ...
            fs, frame_ms, frame_inc_ms);
        
        if (length(act) > 1)
          [m_speech, n_speech] = get_speech_silence_frames(s, start, stop, act);            
          m_speaker_speech = [m_speaker_speech; m_speech];
          n_speaker_samples += n_speech;         
        endif
        
        printf("-");
        fflush(stdout);
        
      endfor
      
    endfor
    
    printf(">\n");
    fflush(stdout);
    
    c_speech{i} = m_speaker_speech;
    v_n_frames_speech(i) = n_speaker_samples;
      
  end
  
  s1 = time();  
  printf("Processing files into speech and silence: %.3f sec.\n", s1 - s0);
  fflush(stdout);

endfunction
