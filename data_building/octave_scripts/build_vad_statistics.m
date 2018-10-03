# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [v_speech_duration] = build_vad_statistics(v_dir_database, ...
  max_speakers = 0)
  
  addpath("E:/1_Proiecte_Curente/1_Speaker_Counting/3rdparty/voicebox");
  pkg load signal
  pkg load ltfat

  % ----------------------------------------------------------------------------
  % Input:
  % ----------------------------------------------------------------------------
  % v_directories - the directory with the input files
  % ----------------------------------------------------------------------------
  % Output:
  % ----------------------------------------------------------------------------
  % v_speech_duration - array with all continuous speech durations

  f_speech_threshold = 0.5;
  n_silence_threshold = 1;
  n_min_speech_length = 1;
  v_directories = glob(strcat(v_dir_database));
  v_speech_duration = [];
  
  n_speakers = length(v_directories);
  if (max_speakers ~= 0)
    if (n_speakers > max_speakers)
      n_speakers = max_speakers;
    endif
  endif  
  
  ##############################################################################
  # Sepparate Silence and Speech
  ##############################################################################
  
  s0 = time();
  
  % Loop across all speaker directories
  for i = 1 : n_speakers
    
    printf("Folder: %s\n", v_directories{i});
    fflush(stdout);
    
    sessions_list = glob(strcat(v_directories{i}, "/*"));
    
    % Loop across all speaker sessions
    for j = 1 : length(sessions_list)
      
      file_list = glob(strcat(sessions_list{j}, "/*.flac"));
      
      % Loop across all speaker recordings
      for k = 1: length(file_list)
        
        % Read speech signal
        s_file = file_list{k};
        [si, fs] = audioread(s_file);
        speech = si(:, 1);
        
        % Apply voice activity detector and tag each sample as speech or silence
        vad_decision = vadsohn(speech, fs, 'nb');        
        % plot(vad_decision(:, 3));
        
        % Received one frame per row
        n_frames = size(vad_decision)(1);
        
        n_first_speech_frame = 0;
        n_last_speech_frame = 0;
        n_silence_frames = 0;
        b_during_silence = 0;
        
        for l = 1 : n_frames
          if (vad_decision(l, 3) < f_speech_threshold)
            n_silence_frames += 1;
            if (n_silence_frames > n_silence_threshold)
              if (b_during_silence == 0)
                b_during_silence = 1;
                if ((n_last_speech_frame ~= 0) && (n_first_speech_frame ~=0))
                  n_speech_length = vad_decision(n_last_speech_frame, 2) ...
                    - vad_decision(n_first_speech_frame, 1);
                  if (n_speech_length > n_min_speech_length)
                    v_speech_duration = [v_speech_duration, n_speech_length / fs];                    
                  endif                 
                endif
              endif
            endif
          else
            if (b_during_silence == 1)
              b_during_silence = 0;
              n_first_speech_frame = l;
            endif
            n_silence_frames = 0;
            n_last_speech_frame = l;           
          endif         
        endfor
        
        % Display progress
        printf("-");
        fflush(stdout);
                
      endfor
    endfor
    
    printf(">\n");
    fflush(stdout);

  end
  
  s1 = time();  
  printf("VAD statistics duration: %.3f sec.\n", s1 - s0);
  fflush(stdout);

endfunction
