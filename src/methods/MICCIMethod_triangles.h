#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <nlohmann/json.hpp>
#include <bitset>
#include "Method.h"
#include "types/IntersectingAreaParameters.h"
#include "types/IntersectingAreaEstimationStrategy.h"
#include "methods/micci/MICCIDescriptor.h"
#include "methods/micci/MICCIGenerator_TriangleMesh.h"
#include "methods/micci/imageDiscretizer.h"
#include <distanceFunctions/RICIDistance.h>
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"

namespace ShapeBench {

    struct MICCIMethod_triangles {


        static void init(const nlohmann::json &config) {
        }

        static void destroy() {}

        static bool usesMeshInput() {
            return true;
        }

        static bool usesPointCloudInput() {
            return false;
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
            Method::throwIncompatibleException();
            return {};
        }

        // Triangle mesh, runs on the CPU
        static ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh &mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            return ShapeBench::micci::generateGrayscaleMICCIDescriptorMultiRadius(mesh, descriptorOrigins, supportRadii);
        }

        // Point cloud, runs on the CPU
        static ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud &cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }


        static __inline__ float computeDescriptorDistance(
                const ShapeBench::TriangleMeshMICCIDescriptor& descriptor,
                const ShapeBench::TriangleMeshMICCIDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeRICIDistance(descriptor, otherDescriptor, earlyExitThreshold);
        }

        __device__ static __inline__ uint64_t computeDescriptorDistanceGPU(
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
            return "MICCI-Triangle";
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["widthPixels"] = spinImageWidthPixels;
            return metadata;
        }
    };
}



