# Standardized Variable Distances (SVD)

[![View Standardized Variable Distances (SVD) on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/84540-standardized-variable-distances-svd)

In this study, a novel machine learning algorithm for multiclass classification is presented. The proposed method is designed based on the Minimum Distance Classifier (MDC) algorithm. The MDC is variance-insensitive because it classifies input vectors by calculating their distances/similarities with respect to class-centroids (average value of input vectors of a class). As it is known, real-world data contains certain proportions of noise. This situation negatively affects the performance of the MDC. To overcome this problem, we developed a variance-sensitive model, which we call Standardized Variable Distances (SVD), considering the standard deviation and z-score (standardized variable) factors.

**Main paper:**

Elen, A., & Avuçlu, E. (2021). Standardized Variable Distances: A distance-based machine learning method. Applied Soft Computing, 98(2021): 106855. doi: https://doi.org/10.1016/j.asoc.2020.106855

https://www.researchgate.net/publication/344947495_Standardized_Variable_Distances_A_distance-based_machine_learning_method

**How to use:**

```matlab
# step1: load a dataset and set train/test variables
data = load('Datasets/wbcd.mat');
% data = load('Datasets/wine.mat');

trainFeatures = data.feature_train;
trainLabels = data.label_train;

testFeatures = data.feature_test;
testLabels = data.label_test;

```

```matlab
# step2: set initial parameters
% Set distance metric; 'mahalanobis', 'seuclidean', 'hamming', 'euclidean'
distanceMetric = 'mahalanobis';

% The number of classes {1,..., n}
classCount = length(unique(trainLabels));

```

```matlab
# step3: train the SVD
resultTrain = funcTrain(trainFeatures, trainLabels, ...
    classCount, distanceMetric);

vectorMean = resultTrain.MeanVectors;
vectorStdDev = resultTrain.StdVectors;

trainRate = resultTrain.CACC;
trainErrors = resultTrain.Misclassification;

```

```matlab
# step4: Print training set statistics
fprintf('Training set statistics\n')
fprintf('-----------------------------------------\n')
fprintf('Learning accuracy: %f\n', trainRate)

for i = 1 : length(trainErrors),
   fprintf('Class #%u errors: %u\n', i, trainErrors(i))
end

fprintf('\n\nConfusion matrix for train\n')
confusionmat(double(trainLabels), double(resultTrain.Predictions))

```


```matlab
# step5: test the SVD
resultTest = funcTest(testFeatures, testLabels, ...
    classCount, distanceMetric, vectorMean, vectorStdDev);

testRate = resultTest.CACC;
testErrors = resultTest.Misclassification;

```

```matlab
# step6: Print test set statistics
fprintf('\n\nTest set statistics\n')
fprintf('-----------------------------------------\n')
fprintf('Classification accuracy: %f\n', testRate)

for i = 1 : length(testErrors),
   fprintf('Class #%u errors: %u\n', i, testErrors(i))
end

fprintf('\nConfusion matrix for test\n')
confusionmat(double(testLabels), double(resultTest.Predictions))

```

**Authors:**

<table border="0" width="90%">
  <tr>
    <td>Abdullah Elen, Assist. Prof.
      
Department of Software Engineering, Faculty of Engineering and Natural Sciences, Bandirma Onyedi Eylul University, 10200 Bandirma, Balikesir/TR

<ul>
<li>E-mail: aelen@bandirma.edu.tr</li>
<li>RG-Profile: https://www.researchgate.net/profile/Abdullah_Elen</li>
<li>Personal Web: https://www.elenium.net</li>
<li>Institutional Web: http://rehber.bandirma.edu.tr/aelen</li>
</ul>
    </td>
  </tr>
  <tr><td></td></tr>
  <tr>
    <td>Emre Avuçlu, Assist. Prof.

Department of Computer Technologies, Vocational School of Tech. Sci., Aksaray University, Aksaray/TR

<ul>
<li>E-mail: emreavuclu@aksaray.edu.tr</li>
<li>RG-Profile: https://www.researchgate.net/profile/Emre-Avuclu</li>
<li>Personal Web: http://www.emreavuclu.com</li>
</ul>
  </td>
  </tr>
</table>
