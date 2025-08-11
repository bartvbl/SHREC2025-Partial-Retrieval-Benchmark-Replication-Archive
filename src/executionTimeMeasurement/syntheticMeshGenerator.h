#pragma once
#include <shapeDescriptor/shapeDescriptor.h>
#include "nlohmann/json.hpp"
#include "benchmarkCore/randomEngine.h"
#include "utils/geometryRandomUtils.h"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

namespace ShapeBench {
    enum class AllowedLocation {
        INSIDE, OUTSIDE
    };

    enum class PointCloudGenerationType {
        UNIFORM_SPREAD, GAUSSIAN_DISTRIBUTED_CLOUDS
    };

    inline glm::mat4 computeSphericalCoordinateTransformation(ShapeDescriptor::cpu::float2 sphericalCoordinates) {
        return glm::rotate(glm::mat4(1.0), sphericalCoordinates.x, glm::vec3(0, 0, 1))
             * glm::rotate(glm::mat4(1.0), sphericalCoordinates.y, glm::vec3(1, 0, 0));
    }

    template<typename DescriptorMethod>
    ShapeDescriptor::cpu::float3 generateRandomCentrePoint(float supportRadius, float meshInstanceScaleFactor, ShapeDescriptor::cpu::float2 meshInstanceOrientationSphericalCoordinates, ShapeBench::randomEngine& engine, AllowedLocation location) {
        ShapeDescriptor::cpu::float3 instanceOrientation = ShapeBench::directionVectorFromSphericalCoordinates(meshInstanceOrientationSphericalCoordinates);
        for(uint32_t attempt = 0; attempt < 10000; attempt++) {

            std::uniform_real_distribution<float> distanceDistribution(-2.0 * supportRadius, 2.0f * supportRadius);
            ShapeDescriptor::cpu::float3 randomPoint = {distanceDistribution(engine), distanceDistribution(engine), distanceDistribution(engine)};


            glm::mat4 boundingBoxTransformation
                    = glm::translate(glm::mat4(1.0), glm::vec3(randomPoint.x, randomPoint.y, randomPoint.z))
                      * computeSphericalCoordinateTransformation(meshInstanceOrientationSphericalCoordinates)
                      * glm::scale(glm::mat4(1.0), glm::vec3(meshInstanceScaleFactor, meshInstanceScaleFactor, meshInstanceScaleFactor));

            bool isAnyCornerInSupportVolume = false;
            bool areAllCornersInSupportVolume = true;

            for(int x = -1; x <= 1; x += 2) {
                for(int y = -1; y <= 1; y += 2) {
                    for(int z = -1; z <= 1; z += 2) {
                        glm::vec4 cornerCoordinate(x, y, z, 1);
                        glm::vec4 transformedCoordinate = boundingBoxTransformation * cornerCoordinate;
                        ShapeDescriptor::cpu::float3 convertedTransformedCoordinate = {transformedCoordinate.x, transformedCoordinate.y, transformedCoordinate.z};
                        ShapeDescriptor::OrientedPoint defaultOrigin;
                        defaultOrigin.vertex = {0, 0, 0};
                        defaultOrigin.normal = {0, 0, 1};
                        bool isInSupportVolume = DescriptorMethod::isPointInSupportVolume(supportRadius, defaultOrigin, convertedTransformedCoordinate);

                        isAnyCornerInSupportVolume = isAnyCornerInSupportVolume || isInSupportVolume;
                        areAllCornersInSupportVolume = areAllCornersInSupportVolume && isInSupportVolume;
                    }
                }
            }

            if(location == AllowedLocation::INSIDE && areAllCornersInSupportVolume) {
                return randomPoint;
            } else if(location == AllowedLocation::OUTSIDE && !isAnyCornerInSupportVolume) {
                return randomPoint;
            }
        }

        throw std::runtime_error("Took too long to generate a valid centre point when generating random mesh!");
    }

    template<typename DescriptorMethod>
    ShapeDescriptor::cpu::Mesh generateSyntheticMesh(const ShapeDescriptor::cpu::Mesh& meshToReplicate,
                                                     uint32_t instanceCount,
                                                     uint32_t meshCountPerChosenLocation,
                                                     float instanceScaleFactor,
                                                     ShapeDescriptor::cpu::float3 meshOrigin,
                                                     ShapeDescriptor::cpu::float2 meshOrientationSphericalCoordinates,
                                                     float supportRadius,
                                                     AllowedLocation location,
                                                     ShapeBench::randomEngine& engine) {

        glm::mat4 descriptorNormalTransformation = computeSphericalCoordinateTransformation(meshOrientationSphericalCoordinates);
        glm::mat4 descriptorPositioningTransformation
            = glm::translate(glm::mat4(1.0), glm::vec3(meshOrigin.x, meshOrigin.y, meshOrigin.z))
            * descriptorNormalTransformation;

        ShapeDescriptor::cpu::Mesh sceneMesh(instanceCount * meshToReplicate.vertexCount);

        if(meshToReplicate.vertexColours != nullptr && sceneMesh.vertexColours == nullptr) {
            sceneMesh.vertexColours = new ShapeDescriptor::cpu::uchar4[sceneMesh.vertexCount];
        }
        if(meshToReplicate.textureCoordinates != nullptr && sceneMesh.textureCoordinates == nullptr) {
            sceneMesh.textureCoordinates = new ShapeDescriptor::cpu::float2[sceneMesh.vertexCount];
        }
        if(meshToReplicate.triangleTextureIDs != nullptr && sceneMesh.triangleTextureIDs == nullptr) {
            sceneMesh.triangleTextureIDs = new uint8_t[sceneMesh.vertexCount / 3];
        }

        ShapeDescriptor::cpu::float2 instanceOrientationSphericalCoordinates = {0, 0};
        ShapeDescriptor::cpu::float3 instancePosition = {0, 0, 0};
        glm::mat4 meshInstanceRotationTransformation(1.0);
        glm::mat4 meshInstanceNormalTransformation(1.0);


        // Duplicate mesh instance several times
        for(uint32_t instanceIndex = 0; instanceIndex < instanceCount; instanceIndex++) {
            uint32_t targetBaseIndex = instanceIndex * meshToReplicate.vertexCount;
            uint32_t indexWithinChosenLocation = instanceIndex % meshCountPerChosenLocation;

            // For the first instance of each location, pick a new location at random
            if(indexWithinChosenLocation == 0) {
                instanceOrientationSphericalCoordinates = ShapeBench::generateRandomSphericalCoordinates(engine);
                instancePosition = ShapeBench::generateRandomCentrePoint<DescriptorMethod>(supportRadius, instanceScaleFactor, instanceOrientationSphericalCoordinates, engine, location);
                meshInstanceRotationTransformation = ShapeBench::computeSphericalCoordinateTransformation(instanceOrientationSphericalCoordinates);
                meshInstanceNormalTransformation = descriptorNormalTransformation * meshInstanceRotationTransformation;
            }

            float specificInstanceScaleFactor = float(indexWithinChosenLocation + 1) / float(meshCountPerChosenLocation);

            glm::mat4 meshInstanceTransformation
                = descriptorPositioningTransformation
                * glm::translate(glm::mat4(1.0), glm::vec3(instancePosition.x, instancePosition.y, instancePosition.z))
                * meshInstanceRotationTransformation
                * glm::scale(glm::mat4(1.0), glm::vec3(specificInstanceScaleFactor, specificInstanceScaleFactor, specificInstanceScaleFactor))
                * glm::scale(glm::mat4(1.0), glm::vec3(instanceScaleFactor, instanceScaleFactor, instanceScaleFactor));

            for(uint32_t vertexIndex = 0; vertexIndex < meshToReplicate.vertexCount; vertexIndex++) {
                const ShapeDescriptor::cpu::float3& instanceVertex = meshToReplicate.vertices[vertexIndex];
                const ShapeDescriptor::cpu::float3& instanceNormal = meshToReplicate.normals[vertexIndex];

                glm::vec4 transformedCoordinate = meshInstanceTransformation * glm::vec4(instanceVertex.x, instanceVertex.y, instanceVertex.z, 1);
                sceneMesh.vertices[targetBaseIndex + vertexIndex] = {transformedCoordinate.x, transformedCoordinate.y, transformedCoordinate.z};

                glm::vec4 transformedNormal = meshInstanceNormalTransformation * glm::vec4(instanceNormal.x, instanceNormal.y, instanceNormal.z, 1);
                sceneMesh.normals[targetBaseIndex + vertexIndex] = {transformedNormal.x, transformedNormal.y, transformedNormal.z};
            }

            // The remainder is not altered and can be copied over directly
            if(meshToReplicate.vertexColours != nullptr) {
                std::copy(meshToReplicate.vertexColours, meshToReplicate.vertexColours + meshToReplicate.vertexCount, sceneMesh.vertexColours + instanceIndex * meshToReplicate.vertexCount);
            }
            if(meshToReplicate.textureCoordinates != nullptr) {
                std::copy(meshToReplicate.textureCoordinates, meshToReplicate.textureCoordinates + meshToReplicate.vertexCount, sceneMesh.textureCoordinates + instanceIndex * meshToReplicate.vertexCount);
            }
            if(meshToReplicate.triangleTextureIDs != nullptr) {
                std::copy(meshToReplicate.triangleTextureIDs, meshToReplicate.triangleTextureIDs + meshToReplicate.vertexCount, sceneMesh.triangleTextureIDs + instanceIndex * meshToReplicate.vertexCount);
            }
        }

        return sceneMesh;
    }
}