#pragma once
// Activation functions gradients
// These functions compute the gradients of various activation functions.   
// The gradients are essential for backpropagation in neural networks, allowing the model to learn from the errors during training.

float identity_gradient(float x);
float binary_step_gradient(float x);
float relu_gradient(float x);
float leakyrelu_gradient(float x, float alpha = 0.01f);
float prelu_gradient(float x, float alpha = 0.01f);
float sigmoid_gradient(float x);        
float tanh_gradient(float x);
float elu_gradient(float x, float alpha = 1.0f);
float softplus_gradient(float x);
float softsign_gradient(float x);
float swish_gradient(float x);
float mish_gradient(float x);
float gelu_gradient(float x);
float gaussian_gradient(float x);
float sinusoid_gradient(float x);