# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_speech_frames, n_speech, m_silence_frames, n_silence] = ...
  get_speech_silence_frames (speech, frame_start, frame_stop, activity)
  
  % Usage: [speech, N_S, silence, N_N] = get_speech_silence_frames(speech, start, stop, activity)
  %
  % This function extracts silence and speech frames and returns 2 containers.
  %
  % Input:
  % 
  % speech - Input speech signal
  % start - Start of frame
  % stop - End of frame
  % activity - Frame tag: speech or silence 
  %
  % Output:
  %
  % m_speech_frames - Speech frames
  % n_speech - Number of speech frames
  % m_silence_frames - silence frames
  % m_silence - Number of silence frames
  
  debug = 0;
  
  m_speech_frames = [];
  m_silence_frames = [];
  n_speech = 0;
  n_silence = 0;
  n = length(activity);
  
  for i = 1 : n - 1
    if (activity(i) == 1)
      frame = speech(frame_start(i) : frame_stop(i));
      m_speech_frames = [m_speech_frames; frame'];
      n_speech ++;
    end
    
    if (activity(i) == 0)
      frame = speech(frame_start(i) : frame_stop(i));
      m_silence_frames = [m_silence_frames; frame'];
      n_silence ++;
    end
  end
  
  if (debug == 1)
    disp 'Speech Frames: ';
    disp sum(activity);
    disp 'Size of Speech Frames Structure: ';
    disp size(m_speech_frames);
    disp 'Size of silence Frames Structure: ';
    disp size(m_silence_frames);   
  end
  
endfunction
