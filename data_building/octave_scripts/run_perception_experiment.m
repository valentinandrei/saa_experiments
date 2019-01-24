listener_id     = randi(1000);
fs              = 16000;
n_max_speakers  = 4;
n_durations     = 4;
n_samples       = 5;
v_frame_ms      = [5000, 2000, 1000, 500];

% -------------------
% Run No Replay Phase
% -------------------

n_total_samples = n_max_speakers * n_durations * n_samples;
m_norep_results = zeros(n_total_samples, 3);
v_sample_ids = randperm(n_total_samples) - 1;

% Load the Mixtures (cMixturesNoReplay)
printf("Se incarca fisierul cu inregistrari.\n");
load mixtures_no_replay.mat

% Start the test
printf("In prima parte a experimentului, nu veti putea reasculta inregistrarile.\n");
printf("Apasati orice tasta pentru a incepe experimentul.\n");
fflush(stdout);
pause();

% DEBUG
n_total_samples = 5;

% Play the sounds
for i = 1 : n_total_samples
  n_index = v_sample_ids(i);
  n_duration_id = 1 + floor(n_index / n_max_speakers / n_samples);
  n_sample_id = 1 + mod(n_index, n_max_speakers * n_samples);
  
  % Extract Mixture
  v_mixture = cMixturesNoReplay{n_duration_id, 1}(n_sample_id, :);
  
  % Extract Label
  v_label = cMixturesNoReplay{n_duration_id, 2}(n_sample_id, :);
  
  % Play Sound and Ask For Count
  n_claimed_speaker_count = 0;
  soundsc(v_mixture, fs); 
    
  printf("\n%d/%d Cati vorbitori simultan ati putut numara? ", i, n_total_samples);
  printf("Introduceti un numar de la 1 la 4, inclusiv.\n");
  fflush(stdout);

  while (n_claimed_speaker_count == 0)
    try
      n_claimed_speaker_count = input("Numar vorbitori (1 - 4): ");
      if ((n_claimed_speaker_count < 1) || (n_claimed_speaker_count > 4))
        n_claimed_speaker_count = 0;
      end
    catch
      n_claimed_speaker_count = 0;
    end_try_catch
  end
  
  % Save answer
  m_norep_results(i, 1) = v_frame_ms(n_duration_id);
  m_norep_results(i, 2) = n_claimed_speaker_count;
  m_norep_results(i, 3) = find(v_label, 1);
  
end

save no_replay.mat m_norep_results

% ----------------
% Run Replay Phase
% ----------------

m_rep_results = zeros(n_total_samples, 3);
v_sample_ids = randperm(n_total_samples) - 1;

% Load the Mixtures (cMixturesNoReplay)
printf("Se incarca fisierul cu inregistrari.\n");
load mixtures_replay.mat

% Start the test
% Start the test
printf("In a doua parte a experimentului VETI PUTEA reasculta inregistrarile.\n");
printf("Apasati orice tasta pentru a incepe experimentul.\n");
fflush(stdout);
pause();

% Play the sounds
for i = 1 : n_total_samples
  n_index = v_sample_ids(i);
  n_duration_id = 1 + floor(n_index / n_max_speakers / n_samples);
  n_sample_id = 1 + mod(n_index, n_max_speakers * n_samples);
  
  % Extract Mixture
  v_mixture = cMixturesReplay{n_duration_id, 1}(n_sample_id, :);
  
  % Extract Label
  v_label = cMixturesReplay{n_duration_id, 2}(n_sample_id, :);
  
  % Play Sound and Ask For Count
  n_claimed_speaker_count = 0;
  
  while (n_claimed_speaker_count == 0)
    soundsc(v_mixture, fs); 
   
    printf("\n%d/%d Cati vorbitori simultan ati putut numara? ", i, n_total_samples);
    printf("Introduceti un numar de la 1 la 4, inclusiv.\n");
    printf("Daca doriti sa reascultati, introduceti orice alt numar.\n\n");
    fflush(stdout);
 
    try
      n_claimed_speaker_count = input("Numar vorbitori (1 - 4): ");
      if ((n_claimed_speaker_count < 1) || (n_claimed_speaker_count > 4))
        n_claimed_speaker_count = 0;
      end
    catch
      n_claimed_speaker_count = 0;
    end_try_catch  
  end
  
  % Save answer
  m_rep_results(i, 1) = v_frame_ms(n_duration_id);
  m_rep_results(i, 2) = n_claimed_speaker_count;
  m_rep_results(i, 3) = find(v_label, 1);
  
end

save with_replay.mat m_rep_results

printf("Experimentul a luat sfarsit. Va multumim.\n");
fflush(stdout);