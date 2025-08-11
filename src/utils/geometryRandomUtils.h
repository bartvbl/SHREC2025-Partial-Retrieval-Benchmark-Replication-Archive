#pragma once

#include "benchmarkCore/randomEngine.h"
#include "shapeDescriptor/shapeDescriptor.h"

namespace ShapeBench {
    ShapeDescriptor::cpu::float3 directionVectorFromSphericalCoordinates(ShapeDescriptor::cpu::float2 sphericalCoordinates);
    ShapeDescriptor::cpu::float2 generateRandomSphericalCoordinates(ShapeBench::randomEngine& engine);
    ShapeDescriptor::cpu::float3 generateRandomDirectionVector(ShapeBench::randomEngine& engine);
}