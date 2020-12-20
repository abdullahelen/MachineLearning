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
% This function is used to compute the absolute z-scores of the features
% based on each class label for the SVD classifier.
%
% Syntax:
% result = funcGetAbsZscores(features, classCount, MeanVectors, StdVectors)
%
% Inputs:
%	features	: training or test data.
%	classCount	: the number of class.
%	MeanVectors	: mean (centroid) vectors.
%   StdVectors	: standard deviation vectors.
%
% Outputs:
%	Absolute z-scores of the features based on each class label.
% *************************************************************************

function result = funcGetAbsZscores(features, classCount, MeanVectors, StdVectors)

    % Gets the number of samples.
    sampleCount = size(features, 1);
    
    result = zeros(sampleCount, classCount);
    
    % Computes absolute z-scores of the features based on each class label.
    % Eq. (14) in the manuscript.
    for j = 1 : sampleCount,
        for i = 1 : classCount,
            result(j, i) = sum(abs(features(j, :) - MeanVectors(i, :)) ...
                ./ StdVectors(i, :));
        end
    end
end