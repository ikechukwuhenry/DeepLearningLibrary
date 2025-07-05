//
//  main.cpp
//  DeepLearningLibrary
//
//  Created by IK on 09/04/2024.
//

#include <iostream>
#include <vector>
#include <numeric>
#include "headers/activation_functions.h"
#include "headers/preprocessing.h"

std::vector<float> vec = { 5,  10, 15, 20, 25, 30, 35, 40,
                        45, 50, 55, 60, 65, 70, 71, 43.2 };

int main(int argc, const char * argv[]) {
    // insert code here...
    float x = 3.892;
    float y = -2.0;
    std::cout << "Hello, World!\n";
    std::cout << "RELU of " << x << " : " << relu(x) << std::endl;
    std::cout << "Sigmoid of " << x << " : " << sigmoid(-x) << std::endl;
    std::cout << "Tanh of " << x << " : " << dllib_tanh(x) << std::endl;
    std::cout << "Leaky RELU of " << y << " : " << leakyrelu(y) << std::endl;
    std::cout << "ELU of " << x << " : " << elu(x) << std::endl;
    return 0;
}
