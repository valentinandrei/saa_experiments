# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_feat_normalized] = do_feature_scaling(m_feat_unnormalized, s_mmm)

  m_mmm   = load(s_mmm);
  v_max   = m_mmm(1, :);
  v_min   = m_mmm(2, :);
  v_mean  = m_mmm(3, :);

  m_feat_normalized  = (m_feat_unnormalized - v_mean) ./ (v_max - v_min);

endfunction
