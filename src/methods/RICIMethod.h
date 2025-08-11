#pragma once

#include "Method.h"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include "distanceFunctions/RICIDistance.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>

namespace ShapeBench {
    struct RICIMethod {
        static void init(const nlohmann::json &config) {

        }

        static void destroy() {

        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const ShapeDescriptor::RICIDescriptor& descriptor,
                const ShapeDescriptor::RICIDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeRICIDistanceGPU<ShapeDescriptor::RICIDescriptor>(descriptor, otherDescriptor, earlyExitThreshold);
        }

        static inline float computeDescriptorDistance(
                const ShapeDescriptor::RICIDescriptor& descriptor,
                const ShapeDescriptor::RICIDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeRICIDistance<ShapeDescriptor::RICIDescriptor>(descriptor, otherDescriptor, earlyExitThreshold);
        }

        static bool usesMeshInput() {
            return true;
        }
        static bool usesPointCloudInput() {
            return false;
        }
        static bool hasGPUKernels() {
            return false;
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            return ShapeDescriptor::generateRadialIntersectionCountImagesMultiRadius(mesh, descriptorOrigins, supportRadii);
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static std::string getName() {
            return "RICI";
        }

        static ShapeBench::IntersectingAreaEstimationStrategy getIntersectingAreaEstimationStrategy() {
            return ShapeBench::IntersectingAreaEstimationStrategy::FAST_CYLINDRICAL;
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            return isPointInCylindricalVolume(descriptorOrigin, supportRadius, supportRadius, samplePoint);
        }

        static double computeIntersectingAreaCustom(const ShapeBench::IntersectionAreaParameters& parameters) {
            Method::throwUnimplementedException();
            return 0;
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["resolution"] = spinImageWidthPixels;
            metadata["distanceFunction"] = "Clutter Resistant Squared Sum";
            return metadata;
        }
    };
}