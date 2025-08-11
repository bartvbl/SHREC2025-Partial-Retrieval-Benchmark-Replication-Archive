

#include "geometryRandomUtils.h"

ShapeDescriptor::cpu::float3 ShapeBench::generateRandomDirectionVector(ShapeBench::randomEngine &engine) {
    ShapeDescriptor::cpu::float2 sphericalCoordinates = generateRandomSphericalCoordinates(engine);
    return directionVectorFromSphericalCoordinates(sphericalCoordinates);
}

ShapeDescriptor::cpu::float2 ShapeBench::generateRandomSphericalCoordinates(ShapeBench::randomEngine &engine) {
    std::uniform_real_distribution<float> uniformDistribution(0, 1);

    float u = uniformDistribution(engine);
    float v = uniformDistribution(engine);

    float theta = 2.0f * float(M_PI) * u;
    float omega = float(std::acos(2.0f * v - 1.0));

    return {theta, omega};
}

ShapeDescriptor::cpu::float3
ShapeBench::directionVectorFromSphericalCoordinates(ShapeDescriptor::cpu::float2 sphericalCoordinates) {
    float theta = sphericalCoordinates.x;
    float omega = sphericalCoordinates.y;

    float sinOmega = std::sin(omega);

    ShapeDescriptor::cpu::float3 direction = {
            std::cos(theta) * sinOmega,
            std::sin(theta) * sinOmega,
            std::cos(omega)
    };

    return direction;
}
