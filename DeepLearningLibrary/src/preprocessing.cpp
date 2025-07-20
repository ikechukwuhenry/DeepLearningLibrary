//
//  preprocessing.cpp
//  DeepLearningLibrary
//
//  Created by IK on 12/04/2024.
//

#include <iostream>
#include <vector>
#include <numeric>

float mean(std::vector<float> vec)
{
    float sum_of_elems = std::accumulate(vec.begin(), vec.end(), 0);
    return sum_of_elems/vec.size();
}
