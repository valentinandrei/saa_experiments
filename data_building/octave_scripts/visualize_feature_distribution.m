function [val_min, val_max] = visualize_feature_distribution (v_feature, n_steps)

  val_min   = min(v_feature);
  val_max   = max(v_feature);
  val_step  = (val_max - val_min) / n_steps;
  v_edges   = val_min : val_step : val_max;
  v_hist    = histc(v_feature, v_edges);
  
  figure();
  bar(v_edges, v_hist);
  grid;

endfunction
