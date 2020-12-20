% *************************************************************************
% Standardized Variable Distances (SVD)
% -------------------------------------------------------------------------
% Programmed by Abdullah Elen, 20.12.2020
% e-mail: aelen@bandirma.edu.tr | https://www.elenium.net
% 
% Reference:
% Elen, A., & Avuclu, E. (2021). Standardized Variable Distances: A
% distance-based machine learning method. Applied Soft Computing, 98, 106855.
% doi: 10.1016/j.asoc.2020.106855
% *************************************************************************

clc;
clear all; %#ok<CLALL>


% Sample datasets was obtained from the UCI machine learning repository
% http://archive.ics.uci.edu/ml/datasets/Wine
% http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)

% Load the WBCD dataset and set train/test variables.
data = load('Datasets/wbcd.mat');

% Load the Wine dataset and set train/test variables.
% data = load('Datasets/wine.mat');

trainFeatures = data.feature_train;
trainLabels = data.label_train;

testFeatures = data.feature_test;
testLabels = data.label_test;

% Set distance metric; 'mahalanobis', 'seuclidean', 'hamming', 'euclidean'
distanceMetric = 'mahalanobis';

% The number of classes {1,..., n}
classCount = length(unique(trainLabels));

% -------------------------------------------------------------------------
% SVD: Training process.
% -------------------------------------------------------------------------
resultTrain = funcTrain(trainFeatures, trainLabels, ...
    classCount, distanceMetric);

vectorMean = resultTrain.MeanVectors;
vectorStdDev = resultTrain.StdVectors;

trainRate = resultTrain.CACC;
trainErrors = resultTrain.Misclassification;


% Print training set statistics.
fprintf('Training set statistics\n')
fprintf('-----------------------------------------\n')
fprintf('Learning accuracy: %f\n', trainRate)

for i = 1 : length(trainErrors),
   fprintf('Class #%u errors: %u\n', i, trainErrors(i))
end

fprintf('\n\nConfusion matrix for train\n')
confusionmat(double(trainLabels), double(resultTrain.Predictions))



% -------------------------------------------------------------------------
% SVD: Test process.
% -------------------------------------------------------------------------
resultTest = funcTest(testFeatures, testLabels, ...
    classCount, distanceMetric, vectorMean, vectorStdDev);

testRate = resultTest.CACC;
testErrors = resultTest.Misclassification;


% Print test set statistics.
fprintf('\n\nTest set statistics\n')
fprintf('-----------------------------------------\n')
fprintf('Classification accuracy: %f\n', testRate)

for i = 1 : length(testErrors),
   fprintf('Class #%u errors: %u\n', i, testErrors(i))
end

fprintf('\nConfusion matrix for test\n')
confusionmat(double(testLabels), double(resultTest.Predictions))
