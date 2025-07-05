//
//  activation_functions.h
//  DeepLearningLibrary
//
//  Created by mac on 09/04/2024.
//

#pragma once

// Activation functions
int identity(int x);
int binary_step(int x);
float relu(float x);
float leakyrelu(float x);
float elu(float x, float alpha = 1.0f);
float prelu(float x, float alpha = 0.01f);
float swish(float x);
float mish(float x);
float gelu(float x);
float gaussian(float x);
float sinusoid(float x);
float sigmoid(float x);
float dllib_tanh(float x);
float softplus(float x, float alpha = 1.0f);
float softsign(float x);
// Add more activation functions as needed