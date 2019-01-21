listener_id     = randi(1000);
fs              = 16000;
n_max_speakers  = 4;
n_durations     = 4;
n_train_samples = 3;
n_test_samples  = 6;

% ------------------
% Run Training Phase
% ------------------

n_total_train = n_max_speakers * n_durations * n_train_samples;
m_training_results = zeros(n_total_train, 2);
v_sample_ids = randperm(n_total_train);

% Load the Mixtures (cMixturesTrain)
load mixtures_train.mat

% Start the test
printf("Apasati orice tasta pentru a incepe experimentul.\n");
pause();

% Play the sounds
for i = 1 : n_total_train
  n_index = v_sample_ids(i);
  n_duration_id = 1 + floor(n_index / n_max_speakers / n_train_samples);
  n_sample_id = 1 + mod(n_index, n_max_speakers * n_train_samples);
  
  % Extract Mixture
  v_mixture = cMixturesTrain{n_duration_id, 1}(n_sample_id, :);
  
  % Play Sound and Ask For Count
  n_claimed_speaker_count = 0;
  
  while (n_claimed_speaker_count == 0)
    soundsc(v_mixture, fs); 
    
    printf("Cati vorbitori simultan ati putut numara?\n");
    printf("Introduceti un numar de la 1 la 4, inclusiv.\n");
    printf("Daca doriti sa reascultati, introduceti orice alt numar.\n");
    fflush(stdout);
 
    n_claimed_speaker_count = input("Numar vorbitori (1 - 4): ");
    if ((n_claimed_speaker_count < 1) || (n_claimed_speaker_count > 4))
      n_claimed_speaker_count = 0;
    end    
  end
  
  % Save answer
  m_training_results(i, 1) = n_index;
  m_training_results(i, 2) = n_claimed_speaker_count;
  
end