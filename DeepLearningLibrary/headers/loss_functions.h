#include <vector>
#include <iostream>

float mean_squared_error(const std::vector<float>& predictions, const std::vector<float>& targets);
float binary_cross_entropy(const std::vector<float>& predictions, const std::vector<float>& targets);   
float mean_absolute_error(const std::vector<float>& predictions, const std::vector<float>& targets);
float categorical_cross_entropy(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets);
float sparse_categorical_cross_entropy(const std::vector<std::vector<float>>& predictions, const std::vector<int>& targets);
float kullback_leibler_divergence(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets);
float hinge_loss(const std::vector<float>& predictions, const std::vector<int>& targets);
float huber_loss(const std::vector<float>& predictions, const std::vector<float>& targets, float delta = 1.0f);