# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_speech_frames, n_speech, m_silence_frames, n_silence] = ...
  get_speech_silence_frames (speech, frame_start, frame_stop, activity)
  
  % This function extracts silence and speech frames and returns 2 containers.
  % ----------------------------------------------------------------------------
  % Input:
  % ----------------------------------------------------------------------------
  % speech    - Input speech signal
  % start     - Start of frame
  % stop      - End of frame
  % activity  - Frame tag: speech or silence 
  % ----------------------------------------------------------------------------
  % Output:
  % ----------------------------------------------------------------------------
  % m_speech_frames   - Speech frames
  % n_speech          - Number of speech frames
  % m_silence_frames  - silence frames
  % m_silence         - Number of silence frames
  
  debug             = 1;  
  n                 = length(activity);
  n_speech          = sum(activity == 1);
  n_silence         = sum(activity == 0);
  i_speech          = 1;
  i_silence         = 1;
  sz_frame          = frame_stop(1) - frame_start(1) + 1;  
  m_speech_frames   = zeros(n_speech, sz_frame);
  m_silence_frames  = zeros(n_silence, sz_frame);
    
  t0 = time();
  
  for i = 1 : n - 1
    % Speech
    if (activity(i) == 1)
      frame = speech(frame_start(i) : frame_stop(i));
      m_speech_frames(i_speech, :) = frame';
      i_speech ++;
    end
    
    % Silence
    if (activity(i) == 0)
      frame = speech(frame_start(i) : frame_stop(i));
      m_silence_frames(i_silence, :) = frame';
      i_silence ++;
    end
  end

  t1 = time();
  printf("speech/silence frames sepparation duration: %.3f seconds\n", t1 - t0);
  fflush(stdout);

  if (debug == 1)
    printf('Speech Frames: %d\n', sum(activity));;
    printf('Size of Speech Frames Structure: %dx%d\n', size(m_speech_frames));    
    printf('Size of silence Frames Structure: %dx%d\n', size(m_silence_frames));
    fflush(stdout);  
  end
  
endfunction