## Author: valentin <valentin@valentin-laptop>
## Created: 2017-04-23

function [pc, sv, n_sv] = do_pca (X)

% [pc,sv,n_sv]  = do_pca(X)
%
% Input:
%
%   x     - Data stored column-vise .
%
% Output:
%
%   pc    - Principal components (eigenvectors of the covariance matrix).
%   sv    - Singular values.
%   n_sv  - Normalized singular values.

C         = cov(X);
[U,D,pc]  = svd(C);
sv        = diag(D);
n_sv      = 100 * sv / sum(sv);

endfunction
