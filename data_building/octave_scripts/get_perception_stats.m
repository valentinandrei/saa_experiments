function [n_volunteers, m_individual_cat_acc, m_aggregated_cat_acc, ...
  m_agg_conf_matrices] = get_perception_stats (s_folder)

n_durations           = 4;
n_speakers            = 4;
n_recs_per_speaker    = 5;
v_frame_sizes         = [500, 1000, 2000, 5000];
v_files               = glob(strcat(s_folder, "/*.mat"));
n_volunteers          = length(v_files);
m_individual_cat_acc  = zeros(n_volunteers, n_durations);
m_aggregated_cat_acc  = zeros(1, n_durations);
m_agg_conf_matrices   = zeros(n_durations, n_speakers, n_speakers);

% Load mat files

for i = 1 : n_volunteers
  s_file              = v_files{i};
  m_struct            = load(s_file);
  m_results           = zeros(n_durations * n_speakers * n_recs_per_speaker, 3);
  m_vol_conf_matrices = zeros(n_durations, n_speakers, n_speakers);
  
  % There are only 2 possible variable names
  try
    m_results = m_struct.m_norep_results;
  catch
    m_results = m_struct.m_rep_results;
  end_try_catch

  for rows = 1 : n_durations * n_speakers * n_recs_per_speaker
    n_duration  = m_results(rows, 1);
    n_claimed   = m_results(rows, 2);
    n_correct   = m_results(rows, 3);
    n_sz_id     = 0;
    
    switch n_duration
      case 500
        n_sz_id = 1;
      case 1000
        n_sz_id = 2;
      case 2000
        n_sz_id = 3;
      case 5000
        n_sz_id = 4;
    end
    
    m_vol_conf_matrices(n_sz_id, n_claimed, n_correct) += 1;
    m_agg_conf_matrices(n_sz_id, n_claimed, n_correct) += 1;
  
  end
  
  for n_sz_id = 1 : 4
    f_speaker_acc = 0;
    for s_id = 1 : n_speakers
      f_speaker_acc += (m_vol_conf_matrices(n_sz_id, s_id, s_id) / n_recs_per_speaker);
    end
    m_individual_cat_acc(i, n_sz_id) = f_speaker_acc / n_speakers;
  end
    
end

for sz_id = 1 : 4
  f_speaker_acc = 0;
  for s_id = 1 : n_speakers
    f_speaker_acc += (m_agg_conf_matrices(sz_id, s_id, s_id) / n_recs_per_speaker / n_volunteers);
  end
  m_aggregated_cat_acc(sz_id) = f_speaker_acc / n_speakers;
end

endfunction
