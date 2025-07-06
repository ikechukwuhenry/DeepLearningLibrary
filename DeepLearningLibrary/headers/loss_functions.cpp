#include <iostream>
#include <cmath>    // For mathematical functions
#include <vector>   
#include <stdexcept> // For exception handling

// Mean Squared Error (MSE) Loss Function
// This function calculates the mean squared error between predictions and targets.
// It assumes that both predictions and targets are vectors of the same size.
// MSE = 1/n * Σ(predictions_i - targets_i)^2 for each sample
// where n is the number of samples, predictions_i is the predicted value for sample i,
// and targets_i is the true value for sample i.
// The function returns the average loss over all samples.
// It throws an exception if the sizes of predictions and targets do not match.
// The function iterates through each prediction and target pair, calculating the squared error
// for each pair and accumulating the result. The final loss is averaged over the number of samples
// The function returns the mean squared error loss.
float mean_squared_error(const std::vector<float>& predictions, const std::vector<float>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }
    
    float mse = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float error = predictions[i] - targets[i];
        mse += error * error;
    }
    
    return mse / predictions.size();
}

// Binary Cross-Entropy Loss Function
// This function calculates the binary cross-entropy loss between predictions and targets.
// It assumes that predictions are probabilities (between 0 and 1) and targets are binary (0 or 1).
// Note: This function does not handle cases where predictions are exactly 0 or 1, which would
// lead to log(0) and result in NaN. In practice, you might want to clip the predictions to avoid this issue.
// BCE = -[y * log(ŷ) + (1-y) * log(1-ŷ)] for each sample
// where y is the target and ŷ is the prediction.
// y is the true label (0 or 1)
// ŷ is the predicted probability (between 0 and 1)
// For multiple samples, you take the average:
// BCE = -1/n * Σ[y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
// The function returns the average loss over all samples.
// It throws an exception if the sizes of predictions and targets do not match or if the values
// of predictions or targets are outside the range [0, 1].
// The function returns the average binary cross-entropy loss.
// It is important to ensure that the predictions are probabilities (between 0 and 1)
// and that the targets are binary (0 or 1).
// The function iterates through each prediction and target pair, calculating the binary cross-entropy
// for each pair and accumulating the result. The final loss is averaged over the number of samples
float binary_cross_entropy(const std::vector<float>& predictions, const std::vector<float>& targets) {

    if (predictions.empty()) {
        throw std::invalid_argument("Predictions and targets cannot be empty.");
    }
    // Check if predictions and targets have the same size
    // If they do not, throw an exception. 
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    } 
    // Check if predictions and targets are in the range [0, 1]
    // This is important to avoid log(0) which is undefined.
    // If predictions or targets are outside this range, throw an exception.
    float bce = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] < 0.0f || predictions[i] > 1.0f) {
            throw std::out_of_range("Predictions must be in the range [0, 1].");
        }
        if (targets[i] < 0.0f || targets[i] > 1.0f) {
            throw std::out_of_range("Targets must be in the range [0, 1].");
        }
        bce += targets[i] * std::log(predictions[i]) + (1 - targets[i]) * std::log(1 - predictions[i]);
    }   
    return -bce / predictions.size();
}

float mean_absolute_error(const std::vector<float>& predictions, const std::vector<float>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }
    
    float mae = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        mae += std::abs(predictions[i] - targets[i]);
    }
    
    return mae / predictions.size();
}

// Loss is used for multiclass classification problems.
// Categorical Cross-Entropy Loss Function
// This function calculates the categorical cross-entropy loss between predictions and targets.
// It assumes that predictions are probabilities for each class and targets are one-hot encoded vectors.
// Categorical Cross-Entropy is defined as:
// L(y, ŷ) = -Σ y_i * log(ŷ_i) for each class i
// where y is the true label (one-hot encoded) and ŷ is the predicted probability distribution.
// For multiple samples, you take the average:
// L(y, ŷ) = -1/n * Σ Σ y_i * log(ŷ_i)
// where n is the number of samples, y_i is the true label for sample i,
// and ŷ_i is the predicted probability for class i.
// n is the number of data points
// k is the number of classes,
// yij is the binary indicator (0 or 1) if class label j is the correct classification for data point i
// ŷij is the predicted probability for class j.
// The function returns the average categorical cross-entropy loss over all samples.

float categorical_cross_entropy(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }
    
    float cce = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i].size() != targets[i].size()) {
            throw std::invalid_argument("Each prediction and target must have the same number of classes.");
        }
        
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            if (targets[i][j] == 1.0f) {
                cce -= std::log(predictions[i][j]);
            }
        }
    }
    
    return cce / predictions.size();
}

// Huber Loss Function
// This function calculates the Huber loss between predictions and targets.
// Huber loss is less sensitive to outliers than squared error loss.
// It is defined as:
// L_delta(y, f(x)) = 0.5 * (y - f(x))^2 if |y - f(x)| <= delta
// L_delta(y, f(x)) = delta * (|y - f(x)| - 0.5 * delta) if |y - f(x)| > delta
// where y is the target, f(x) is the prediction, and delta is a threshold parameter.
// The function returns the average Huber loss over all samples.
// It throws an exception if the sizes of predictions and targets do not match.
float huber_loss(const std::vector<float>& predictions, const std::vector<float>& targets, float delta = 1.0f) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }
    
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float error = predictions[i] - targets[i];
        if (std::abs(error) <= delta) {
            loss += 0.5f * error * error; // Quadratic loss
        } else {
            loss += delta * (std::abs(error) - 0.5f * delta); // Linear loss
        }
    }
    
    return loss / predictions.size();
}

// Sparse Categorical Cross-Entropy Loss Function
// This function calculates the sparse categorical cross-entropy loss between predictions and targets.
// It is used when the targets are integers representing class indices.
// The function assumes that predictions are probabilities for each class and targets   // are integers representing the true class index.
// It is defined as:
// L(y, ŷ) = -Σ log(ŷ_i * y) for each sample i
// where y is the true class index and ŷ_i is the predicted probability for class i
// The function returns the average sparse categorical cross-entropy loss over all samples.
// It throws an exception if the sizes of predictions and targets do not match or if the targets are out of range.
// The function iterates through each prediction and target pair, calculating the sparse
// categorical cross-entropy for each pair and accumulating the result. The final loss is averaged over the number of samples.
// The function returns the average sparse categorical cross-entropy loss.
// It is important to ensure that the predictions are probabilities (between 0 and 1)
// and that the targets are valid class indices (non-negative integers less than the number of classes
float sparse_categorical_cross_entropy(const std::vector<std::vector<float>>& predictions, const std::vector<int>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }
    
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (targets[i] < 0 || targets[i] >= predictions[i].size()) {
            throw std::out_of_range("Target index is out of range for predictions.");
        }
        loss -= std::log(predictions[i][targets[i]]);
    }
    
    return loss / predictions.size();
}

// Kullback-Leibler Divergence Loss Function
// This function calculates the Kullback-Leibler divergence loss between two probability distributions.
// It is defined as:
// KL(p || q) = Σ p_i * log(p_i / q_i)
// where p is the true distribution and q is the predicted distribution.
// The function returns the average Kullback-Leibler divergence loss over all samples.
// It throws an exception if the sizes of predictions and targets do not match or if the values
// of predictions or targets are outside the range [0, 1].
float kullback_leibler_divergence(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }           
    float kl_div = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i].size() != targets[i].size()) {
            throw std::invalid_argument("Each prediction and target must have the same number of classes.");
        }   
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            if (predictions[i][j] < 0.0f || predictions[i][j] > 1.0f) {
                throw std::out_of_range("Predictions must be in the range [0, 1].");
            }
            if (targets[i][j] < 0.0f || targets[i][j] > 1.0f) {
                throw std::out_of_range("Targets must be in the range [0, 1].");
            }
            if (targets[i][j] == 0.0f) continue; // Avoid log(0)
            kl_div += targets[i][j] * std::log(targets[i][j] / predictions[i][j]);
        }
    }
    return kl_div / predictions.size();
}

// Hinge Loss Function
// This function calculates the hinge loss for binary classification problems.          
// Hinge loss is commonly used for "maximum-margin" classification, most notably for support vector machines.
// It is defined as:
// L(y, f(x)) = max(0, 1 - y * f(x))
// where y is the true label (-1 or 1) and f(x) is the predicted        
// score (not a probability).
// The function returns the average hinge loss over all samples.
// It throws an exception if the sizes of predictions and targets do not match or if the targets
// are not -1 or 1.
float hinge_loss(const std::vector<float>& predictions, const std::vector<int>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }   
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (targets[i] != -1 && targets[i] != 1) {
            throw std::invalid_argument("Targets must be -1 or 1.");
        }
        float margin = 1.0f - targets[i] * predictions[i];  
        if (margin > 0) {
            loss += margin; // Only add positive margins
        }
    }
    return loss / predictions.size();
}