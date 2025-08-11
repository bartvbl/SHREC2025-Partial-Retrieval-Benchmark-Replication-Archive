#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

inline __host__ __device__ float Gaussian(float mean, float stddev, float value) {
    float exponent = value - mean;
    exponent *= -exponent;
    exponent /= 2 * stddev * stddev;

    float result = std::exp(exponent);
    result /= stddev * std::sqrt(2.0f * M_PI);

    return result;
}

inline __host__ float Guassian2D(ShapeDescriptor::cpu::float2 mean, float stddev, ShapeDescriptor::cpu::float2 point) {
    return Gaussian(mean.x, stddev, point.x) * Gaussian(mean.y, stddev, point.y);
}

inline __device__ float Guassian2D(float2 mean, float stddev, float2 point) {
    return Gaussian(mean.x, stddev, point.x) * Gaussian(mean.y, stddev, point.y);
}