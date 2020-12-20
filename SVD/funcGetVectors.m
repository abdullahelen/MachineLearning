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
% This function is used to compute the mean (centroid) and standard
% deviation vectors for the SVD classifier.
%
% Syntax:
% result = funcGetVectors(features, labels)
%
% Inputs:
%	features	: training data (MxN).
%	labels      : class labels corresponding to training data (Mx1).
%	classCount	: the number of class.
%	distMetric	: distance metric 'euclidean', 'cityblock', etc.
%
% Outputs:
%	MeanVectors	: mean (centroid) vectors.
%   StdVectors	: standard deviation vectors.
% *************************************************************************

function result = funcGetVectors(features, labels)

    % The number of classes {1 ... n}
    classCount = length(unique(labels));

    % Size of the feature space.
    featureCount = size(features, 2);

    % Preallocates mean (centroid) and standard deviation vectors.
    result.MeanVectors = zeros(classCount, featureCount);
    result.StdVectors = zeros(classCount, featureCount);

    % Computes mean (centroid) and standard deviation vectors.
    for i = 1 : classCount,
        result.MeanVectors(i, :) = mean(features(labels == i, :));
        result.StdVectors(i, :) = std(features(labels == i, :));
    end
end