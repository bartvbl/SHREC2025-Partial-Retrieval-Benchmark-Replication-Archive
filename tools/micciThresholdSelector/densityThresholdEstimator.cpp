#include <shapeDescriptor/shapeDescriptor.h>
#include "densityThresholdEstimator.h"
#include "fmt/format.h"
#include <fmt/ranges.h>
#include "methods/micci/MICCIGenerator_TriangleMesh.h"
#include "methods/micci/MICCIGenerator_PointCloud.h"

template<typename DescriptorType, typename PixelType>
void computePixelStatistics(const DescriptorType &image, PixelType& maxValue, PixelType& totalSum, float& nonzeroPixelAverage) {
    maxValue = 0;
    totalSum = 0;
    nonzeroPixelAverage = 0;
    uint32_t count = 0;
    for(PixelType content : image.contents) {
        maxValue = std::max(maxValue, content);
        totalSum += content;
        count += (content != 0 ? 1 : 0);
        nonzeroPixelAverage += float(content != 0 ? ((float(content) - float(nonzeroPixelAverage)) / double(count)) : 0);
    }
}
/*
ShapeBench::SimilarityStatistics ShapeBench::computeAverageSimilarity(
        const ShapeDescriptor::cpu::array<ShapeBench::PointCloudMICCIDescriptor> pointCloudDescriptors,
        const ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> triangleDescriptors,
        const float threshold) {
    const size_t imageCount = pointCloudDescriptors.length;

    ShapeBench::SimilarityStatistics stats;
    stats.thresholdUsed = threshold;
    stats.minSimilarity = std::numeric_limits<float>::max();
    stats.averageSimilarity = 0;
    stats.maxSimilarity = 0;

    for(size_t image = 0; image < imageCount; image++) {
        const ShapeBench::MICCIDescriptor sampleDescriptor = ShapeBench::micci::discretiseMultiModalImage(nonDiscreteDescriptors[image], threshold);
        const ShapeBench::MICCIDescriptor& referenceDescriptor = groundTruthDescriptors.content[image];

        float similarity = ShapeBench::computeImageSimilarity(sampleDescriptor, referenceDescriptor);

        if(std::isnan(similarity)) {
            similarity = 0;
        }

        stats.minSimilarity = std::min<float>(stats.minSimilarity, similarity);
        stats.maxSimilarity = std::max<float>(stats.maxSimilarity, similarity);
        stats.averageSimilarity += float(similarity - stats.averageSimilarity) / float(image + 1);
    }

    return stats;
}
*/
ShapeBench::SimilarityStatistics ShapeBench::estimateOptimalDensityThresholds(
        ShapeDescriptor::cpu::Mesh mesh,
        ShapeDescriptor::cpu::PointCloud pointCloud,
        ShapeDescriptor::OrientedPoint origin,
        float supportRadius,
        std::filesystem::path outputFileDirectory) {

    std::vector<float> supportRadii(1, supportRadius);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> origins {1, &origin};

    ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> triangleDescriptors
        = ShapeBench::micci::generateGrayscaleMICCIDescriptorMultiRadius(mesh, origins, supportRadii);

    ShapeDescriptor::cpu::array<ShapeBench::PointCloudMICCIDescriptor> pointCloudDescriptors
        = ShapeBench::micci::generateGrayscaleMICCIDescriptorsMultiRadius(pointCloud, origins, supportRadii);


    float maxPointCloudPixelValue = 0;
    float pointCloudTotalPixelSum = 0;
    float pointCloudNonzeroPixelAverage = 0;
    //computePixelStatistics(pointCloudDescriptors[0], maxPointCloudPixelValue, pointCloudTotalPixelSum, pointCloudNonzeroPixelAverage);

    uint32_t triangleTotalPixelSum = 0;
    float triangleNonzeroPixelAverage = 0;
    uint32_t maxTrianglePixelValue = 0;
    //computePixelStatistics(triangleDescriptors[0], maxTrianglePixelValue, triangleTotalPixelSum, triangleNonzeroPixelAverage);

    uint32_t RICITotalPixelSum = 0;
    float RICINonzeroPixelAverage = 0;
    uint32_t maxRICIPixelValue = 0;
    //computePixelStatistics(groundTruthDescriptors[0], maxRICIPixelValue, RICITotalPixelSum, RICINonzeroPixelAverage);

    for(uint32_t pixel = 0; pixel < spinImageWidthPixels * spinImageWidthPixels; pixel++) {
        if(triangleDescriptors[0].contents[pixel] != 0) {
            std::cout << pointCloudDescriptors[0].contents[pixel] / triangleDescriptors[0].contents[pixel] << ", " << pointCloudDescriptors[0].contents[pixel] << ", " << triangleDescriptors[0].contents[pixel] << std::endl;
        }
    }

    ShapeDescriptor::writeDescriptorImages({1, reinterpret_cast<ShapeDescriptor::SpinImageDescriptor*>(pointCloudDescriptors.content)}, "pointCloud.png", false, 1);
    ShapeDescriptor::writeDescriptorImages({1, reinterpret_cast<ShapeDescriptor::RICIDescriptor*>(triangleDescriptors.content)}, "triangle.png", false, 1);

    int dontcare;
    std::cout << "Picture done";
    std::cin >> dontcare;

    ShapeBench::SimilarityStatistics bestStatistics;


    std::array<float, 100> distances {};

    //fmt::print("{}, {}, {}, {}, {}, {}, {}, {}, {}\n", maxPointCloudPixelValue, pointCloudTotalPixelSum, pointCloudNonzeroPixelAverage,
    //                                                   maxTrianglePixelValue,   triangleTotalPixelSum,   triangleNonzeroPixelAverage,
    //                                                   maxRICIPixelValue,       RICITotalPixelSum,       RICINonzeroPixelAverage);

    fmt::println("{}", maxPointCloudPixelValue / float(maxRICIPixelValue));

    for(uint32_t i = 0; i < distances.size(); i++) {
        //fmt::print("{}\n", distances);

    }

    /*
    size_t thresholdMin = 1;
    float thresholdStep = ShapeBench::micci::SAMPLES_PER_AXIS / 2.0;
    size_t thresholdLimit = size_t(maxPixelValue / thresholdStep);

    const size_t thresholdCount = thresholdLimit - thresholdMin;
    std::vector<ShapeDescriptor::SimilarityStatistics> statistics(thresholdCount);

    // Try a range of thresholds
    #pragma omp parallel for schedule(dynamic)
    for (size_t thresholdIndex = thresholdMin; thresholdIndex < thresholdLimit; thresholdIndex++) {
        float threshold = float(thresholdIndex) * thresholdStep;
        statistics.at(thresholdIndex - thresholdMin) = ShapeDescriptor::computeAverageSimilarity(sampleDescriptors, referenceDescriptors, threshold);
    }

    size_t bestThresholdIndex = 0;


    for (size_t thresholdIndex = 0; thresholdIndex < thresholdCount; thresholdIndex++) {
        ShapeBench::SimilarityStatistics stats = statistics.at(thresholdIndex);
        if(stats.averageSimilarity > bestStatistics.averageSimilarity) {
            bestStatistics = stats;
            bestThresholdIndex = thresholdIndex;
        }
    }

    std::cout << "Best threshold: " << bestStatistics.thresholdUsed << std::endl;
    std::cout << "Best threshold - average similarity: " << bestStatistics.averageSimilarity << std::endl;
    std::cout << "Best threshold - worst similarity: " << bestStatistics.minSimilarity << std::endl;
    std::cout << "Best threshold - best similarity: " << bestStatistics.maxSimilarity << std::endl;


    // Fine tune best threshold
    double step = ShapeBench::micci::SAMPLES_PER_AXIS / 2.0;
    for(int iteration = 0; iteration < fineTuningIterationCount; iteration++) {
        double previousSimilarity = bestStatistics.averageSimilarity;
        double previousThreshold = bestStatistics.thresholdUsed;

        double lowThreshold = bestStatistics.thresholdUsed - step;
        double highThreshold = bestStatistics.thresholdUsed + step;

        ShapeBench::SimilarityStatistics lowStats = ShapeBench::computeAverageSimilarity(sampleDescriptors, referenceDescriptors, lowThreshold);
        ShapeBench::SimilarityStatistics midStats = bestStatistics;
        ShapeBench::SimilarityStatistics highStats = ShapeBench::computeAverageSimilarity(sampleDescriptors, referenceDescriptors, highThreshold);

        if (midStats.averageSimilarity >= lowStats.averageSimilarity && midStats.averageSimilarity >= highStats.averageSimilarity) {
            bestStatistics = midStats;
        } else if (lowStats.averageSimilarity >= midStats.averageSimilarity && lowStats.averageSimilarity >= highStats.averageSimilarity) {
            bestStatistics = lowStats;
        } else if (highStats.averageSimilarity >= lowStats.averageSimilarity && highStats.averageSimilarity >= midStats.averageSimilarity) {
            bestStatistics = highStats;
        } else {
            throw std::runtime_error("This should be mathematically impossible.");
        }

        assert(bestStatistics.averageSimilarity >= previousSimilarity);

        step /= 2.0;
    }

    ShapeDescriptor::cpu::array<ShapeBench::micci::> discreteDescriptors(sampleDescriptors.length);
    for(int i = 0; i < discreteDescriptors.length; i++) {
        discreteDescriptors[i] = ShapeDescriptor::internal::discretiseMultiModalImage(sampleDescriptors[i], bestStatistics.thresholdUsed);
    }

    if(!outputFileDirectory.empty()) {
        std::filesystem::create_directories(outputFileDirectory);
        std::string uniqueString = ShapeDescriptor::generateUniqueFilenameString();
        ShapeDescriptor::dump::descriptors(referenceDescriptors, outputFileDirectory / ("reference_descriptors_" + uniqueString + ".png"), 50, 2000);
        ShapeDescriptor::dump::descriptors(discreteDescriptors, outputFileDirectory / ("images_discrete_" + uniqueString + "_" + std::to_string(device_pointCloud.pointCount) + "_descriptors.png"), 50, 2000);
        ShapeDescriptor::dump::descriptors(sampleDescriptors, outputFileDirectory / ("images_sample_" + uniqueString + "_" + std::to_string(device_pointCloud.pointCount) + "_descriptors.png"), false, 50, 2000);
        ShapeDescriptor::dump::descriptorComparisonImage(outputFileDirectory / ("images_comparison_" + uniqueString + "_" + std::to_string(device_pointCloud.pointCount) + "_descriptors.png"),
                                                         referenceDescriptors, discreteDescriptors, {0, nullptr}, 50, 2000);
    }



     */

    ShapeDescriptor::free(triangleDescriptors);
    ShapeDescriptor::free(pointCloudDescriptors);

    return bestStatistics;
}
