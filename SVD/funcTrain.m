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
% 
% Description:
% This function is used to train the SVD according to the specified
% distance metric.
%
% Syntax:
% result = funcTrain(features, labels, classCount, distanceMetric)
%
% Inputs:
%	features	: training data (MxN).
%	labels     : class labels corresponding to training data (Mx1).
%	classCount	: the number of class.
%	distMetric	: distance metric 'euclidean', 'cityblock', etc.
%
% Outputs:
%	MeanVectors       : mean (centroid) vectors.
%   StdVectors        : standard deviation vectors.
%   Misclassification : the number of misclassification for each class.
%   CACC              : learning/classification accuracy.
%   Predictions       : learning/prediction results.
% *************************************************************************

function result = funcTrain(features, labels, classCount, distMetric)

    % Computes mean (centroid) and standard deviation vectors.
    % Eq. (11) and (13) in the manuscript.
    vectors = funcGetVectors(features, labels);
    
    % Set result
    result.MeanVectors = vectors.MeanVectors;
    result.StdVectors = vectors.StdVectors;
    
    % Computes distances between every sample and each of the mean vector.
    % Eq. (12) in the manuscript.
    distances = pdist2(features, vectors.MeanVectors, distMetric);
    
    % Computes absolute z-scores of the features based on each class label.
    % Eq. (14) in the manuscript.
    absZscores = funcGetAbsZscores(...
        features, classCount, vectors.MeanVectors, vectors.StdVectors);
    
    % Gets the SVD scores for each input vector.
    % Eq. (15) in the manuscript.
    svd = absZscores .* distances;
    
    % Gets the number of samples in the train set.
    sampleCount = length(features);

    % Preallocates the per class misclassification counts array.
    misclassification = zeros(1, classCount);
    
    predictions = zeros(sampleCount, 1);

    % For each the SVD vector, increment misclassification count if an
    % incorrect classification occurs.
    for i = 1 : sampleCount,
        indx = find(svd(i, :) == min(svd(i, :)));
        predictions(i) = indx;
        misclassification(labels(i)) = ...
            misclassification(labels(i)) + (indx ~= labels(i));
    end

    % Set result.
    result.Predictions = predictions;
    result.Misclassification = misclassification;
    
    % Compute classification accuracy.
    result.CACC = 1.0  - (sum(misclassification) / sampleCount);
end