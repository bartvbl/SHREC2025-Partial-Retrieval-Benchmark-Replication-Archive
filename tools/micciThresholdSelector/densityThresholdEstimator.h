#pragma once

#include <vector>
#include <filesystem>
#include "methods/micci/MICCIDescriptor.h"

namespace ShapeBench {
    struct DensityThresholdEstimate {
        float bestSampleThreshold = 0;
        float similarityToGroundTruth = 0;
        float debug_average = 0;
    };

    struct SimilarityStatistics {
        float minSimilarity = 0;
        float maxSimilarity = 0;
        float averageSimilarity = 0;
        float thresholdUsed = 0;
    };

    SimilarityStatistics computeAverageSimilarity(
            ShapeDescriptor::cpu::array<ShapeBench::PointCloudMICCIDescriptor> pointCloudDescriptors,
            ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> triangleDescriptors,
            float threshold);

    SimilarityStatistics estimateOptimalDensityThresholds(
            ShapeDescriptor::cpu::Mesh mesh,
            ShapeDescriptor::cpu::PointCloud pointCloud,
            ShapeDescriptor::OrientedPoint origin,
            float supportRadius,
            std::filesystem::path outputFileDirectory = "");
}