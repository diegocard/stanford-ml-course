function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

potential_C = [0.01 0.03 0.1 0.3 1 3 10 30];
potential_S = [0.01 0.03 0.1 0.3 1 3 10 30];
minimum_error = inf;

for i=1:length(potential_C)
  for j=1:length(potential_S)
    current_C = potential_C(i);
    current_S = potential_S(j);
    model = svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_S)); 
    predictions = svmPredict(model, Xval);
    predictions_error = mean(double(predictions ~= yval));
    if (predictions_error < minimum_error)
      minimum_error = predictions_error;
      C = current_C;
      S = current_S;
    endif
  end
end

% Best value found for C = 1
% Best value fonund for sigma = 0.1

% =========================================================================

end
