## Author: valentin <valentin@valentin-laptop>
## Created: 2017-05-06

function [v_histogram] = get_histogram (v_signal, f_start, f_stop, n_bins)

f_step = (f_stop - f_start) / n_bins;
v_edges = f_start : f_step : f_stop;
v_histogram = histc(v_signal, v_edges);
v_histogram = v_histogram / length(v_signal);

endfunction
