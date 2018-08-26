function [X_norm, mu, sigma] = do_feature_normalization(X)

X_norm  = X;
mu      = zeros(1, size(X, 2));
sigma   = zeros(1, size(X, 2));
mu      = mean(X);
sigma   = std(X);

for i = 1 : size(X, 2)
  XminusMu = X(:, i) - mu(i);
  X_norm(:, i) = XminusMu / sigma(i);
end

end
