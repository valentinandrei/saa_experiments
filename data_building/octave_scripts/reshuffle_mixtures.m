load mixtures_no_replay_mat7.mat
load mixtures_replay_mat7.mat

n_frames              = 4;
cMixturesNoReplay_new = cell(n_frames, 2);
cMixturesReplay_new   = cell(n_frames, 2);

for i = 1 : n_frames
  
  m_mix_no_rep  = cMixturesNoReplay{i, 1};
  v_lab_no_rep  = cMixturesNoReplay{i, 2};
  m_mix_w_rep   = cMixturesReplay{i, 1};
  v_lab_w_rep   = cMixturesReplay{i, 2};
  
  v_target  = [0, 0, 0, 0];
  m_mix     = [m_mix_no_rep; m_mix_w_rep;
  v_lab     = [v_lab_no_rep; v_lab_w_rep;  
  
  m_mix_no_rep_new = [];
  v_lab_no_rep_new = [];
  
  
endfor
