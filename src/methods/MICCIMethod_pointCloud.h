#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <nlohmann/json.hpp>
#include <bitset>
#include "Method.h"
#include "types/IntersectingAreaParameters.h"
#include "types/IntersectingAreaEstimationStrategy.h"
#include "methods/micci/MICCIDescriptor.h"
#include "methods/micci/MICCIGenerator_TriangleMesh.h"
#include "benchmarkCore/common-procedures/pointCloudSampler.h"
#include "methods/micci/MICCIGenerator_PointCloud.h"
#include "methods/micci/imageDiscretizer.h"
#include "distanceFunctions/RICIDistance.h"
#include "methods/micci/MICCIGenerator_PointCloud_GPU.h"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"

namespace ShapeBench {

    static float levelThresholdPerUnitArea;

    struct MICCIMethod_pointCloud {


        static void init(const nlohmann::json &config) {
            float measuredThreshold = readDescriptorConfigValue<float>(config, getName(), "levelThreshold");
            float supportRadiusUsed = readDescriptorConfigValue<float>(config, getName(), "forSupportRadius");
            float area = supportRadiusUsed * supportRadiusUsed;
            levelThresholdPerUnitArea = measuredThreshold / area;
        }
        static void destroy() {}

        static bool usesMeshInput() {
            return false;
        }

        static bool usesPointCloudInput() {
            return true;
        }

        static bool hasGPUKernels() {
            return false;
        }

        static ShapeDescriptor::gpu::array<ShapeBench::TriangleMeshMICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh &mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> &device_descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        // Point cloud, runs on the GPU
        static ShapeDescriptor::gpu::array<ShapeBench::TriangleMeshMICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud &cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> &device_descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {

            ShapeDescriptor::gpu::array<float> device_supportRadii(supportRadii.size());
            checkCudaErrors(cudaMemcpy(device_supportRadii.content, supportRadii.data(), sizeof(float) * supportRadii.size(), cudaMemcpyHostToDevice));

            ShapeDescriptor::gpu::array<ShapeBench::PointCloudMICCIDescriptor> grayscaleDescriptors = ShapeBench::micci::generateGrayscaleMICCIDescriptorsMultiRadiusGPU(cloud, device_descriptorOrigins, device_supportRadii);
            ShapeDescriptor::gpu::array<ShapeBench::TriangleMeshMICCIDescriptor> discretisedDescriptors = ShapeBench::micci::discretiseMICCIImages(grayscaleDescriptors, device_supportRadii, levelThresholdPerUnitArea);
            ShapeDescriptor::free(grayscaleDescriptors);
            ShapeDescriptor::free(device_supportRadii);
            return discretisedDescriptors;
        }

        // Triangle mesh, runs on the CPU
        static ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh &mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {

            Method::throwIncompatibleException();
            return {};
        }

        // Point cloud, runs on the CPU
        static ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud &cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            ShapeDescriptor::cpu::array<ShapeBench::PointCloudMICCIDescriptor> grayscaleDescriptors = ShapeBench::micci::generateGrayscaleMICCIDescriptorsMultiRadius(cloud, descriptorOrigins, supportRadii);
            ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> converted(grayscaleDescriptors.length);

            for(uint32_t i = 0; i < grayscaleDescriptors.length; i++) {
                float supportRadius = supportRadii.at(i);
                float planeArea = supportRadius * supportRadius;
                float radiusAdjustedThreshold = planeArea * levelThresholdPerUnitArea;
                converted.content[i] = ShapeBench::micci::discretiseMICCIImage(grayscaleDescriptors.content[i], radiusAdjustedThreshold);
            }

            ShapeDescriptor::free(grayscaleDescriptors);

            return converted;
        }


        static __inline__ float computeDescriptorDistance(
                const ShapeBench::TriangleMeshMICCIDescriptor& descriptor,
                const ShapeBench::TriangleMeshMICCIDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeRICIDistance(descriptor, otherDescriptor, earlyExitThreshold);
        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const ShapeBench::TriangleMeshMICCIDescriptor& descriptor,
                const ShapeBench::TriangleMeshMICCIDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeRICIDistanceGPU(descriptor, otherDescriptor, earlyExitThreshold);
        }

        static ShapeBench::IntersectingAreaEstimationStrategy getIntersectingAreaEstimationStrategy() {
            return ShapeBench::IntersectingAreaEstimationStrategy::FAST_CYLINDRICAL;
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            return ShapeBench::isPointInCylindricalVolume(descriptorOrigin, supportRadius, supportRadius, samplePoint);
        }

        static double computeIntersectingAreaCustom(const ShapeBench::IntersectionAreaParameters& parameters) {
            Method::throwUnimplementedException();
            return 0;
        }

        static std::string getName() {
            return "MICCI-PointCloud";
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["widthPixels"] = spinImageWidthPixels;
            metadata["thresholdPerUnitArea"] = levelThresholdPerUnitArea;
            return metadata;
        }
    };
}



