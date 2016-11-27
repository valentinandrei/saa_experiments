# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [] = analyze_dataset (file_data, file_labels)

  % Usage:
  %
  % This function analyzes an input dataset
  %
  % Input:
  %
  % file_data   - Input data file
  % file_lables - Labels data file
  
  x = load(file_data);
  y = load(file_labels);
  
  if (size(x, 1) != length(y))
    disp('Input files do not correspond!');
  end
  
  nInputs = length(y);
  nMaxSpeakers = max(y);
  vSamples = zeros(nMaxSpeakers, 1);
  vMean = zeros(nMaxSpeakers, 1);
  
  % Compute Mean
  vMean = mean(x')';
  
  figure(1);
  plot(vMean); grid;
  xlabel('Sample'); ylabel('Mean');
  
  figure(2);
  for i = 0 : nMaxSpeakers
    vSpeakerSamples = (y == i);
    vTempMean = vSpeakerSamples .* vMean;
    vIndexes = find(vTempMean != 0);
    vSpeakerMean = vTempMean(vIndexes);
    
    subplot(nMaxSpeakers + 1, 1, i+1); plot(vSpeakerMean); grid;
    xlabel(mean(vSpeakerMean)); ylabel('Mean');
  end

endfunction