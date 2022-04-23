% *************************************************************************
% Adaptive Gaussian Kernel
% In this study, an adaptive kernel is proposed based on the Gaussian
% function, which is used in Support Vector Machine (SVM). While the sigma
% parameter is determined as an arbitrary value in the traditional Gaussian
% kernel, the proposed method calculates an adaptive value depending on the
% input vectors.
% *************************************************************************

function result = AdaptiveGaussianKernel(U, V)
	% The Euclidean norm of the input vectors.
	normEuclidean = sqrt(sum((U - V).^2, 2));
	normSquare = normEuclidean.^2;
	
	stdDev = std(normSquare);
    
    normSquare = normSquare - stdDev;
	
	% Find minimum value.
	minVal = min(normSquare);
    
    if (minVal < 0)
		% offset operation
        A = normSquare + abs(minVal);
    else
        A = normSquare;
    end
    
    result = exp(-A / (stdDev + eps));
end
