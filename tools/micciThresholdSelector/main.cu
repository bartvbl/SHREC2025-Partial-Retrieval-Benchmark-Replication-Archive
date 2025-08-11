

#include <filesystem>
#include "arrrgh.hpp"
#include "nlohmann/json.hpp"
#include "benchmarkCore/config/ConfigReader.h"
#include "dataset/Dataset.h"
#include "dataset/DatasetLoader.h"
#include "benchmarkCore/common-procedures/pointCloudSampler.h"
#include "methods/MICCIMethod_triangles.h"
#include "benchmarkCore/randomEngine.h"
#include "benchmarkCore/common-procedures/meshLoader.h"
#include "densityThresholdEstimator.h"
#include "fmt/core.h"
#include "Gaussian2D.h"
#include "methods/micci/MICCIGenerator_PointCloud.h"
#include "utils/prettyprint.h"
#include "distanceFunctions/RICIDistance.h"

uint64_t sumOfSquaredDifferences(const ShapeBench::TriangleMeshMICCIDescriptor& descriptorA,
                                 const ShapeBench::TriangleMeshMICCIDescriptor& descriptorB) {
    uint64_t sumOfDifferences = 0;
    for(uint32_t i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
        uint32_t pixelA = descriptorA.contents[i];
        uint32_t pixelB = descriptorB.contents[i];
        uint32_t delta = pixelB > pixelA ? pixelB - pixelA : pixelA - pixelB;
        sumOfDifferences += delta * delta;
    }
    return sumOfDifferences;
}

int main(int argc, const char** argv) {
    const std::string replicationDisabledString = "UNSPECIFIED";

    arrrgh::parser parser("shapebench", "Benchmark tool for 3D local shape descriptors");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message", 'h', arrrgh::Optional, false);
    const auto& configurationFile = parser.add<std::string>(
            "configuration-file", "Location of the file from which to read the experimental configuration", '\0', arrrgh::Required, "../cfg/config.json");

    try {
        parser.parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }


    // --- Compute experiment configuration ---
    std::cout << "Reading configuration.." << std::endl;
    const std::filesystem::path configurationFileLocation(configurationFile.value());
    bool configurationFileExists = std::filesystem::exists(configurationFileLocation);
    nlohmann::json config;

    if(!configurationFileExists) {
        throw std::runtime_error("The specified configuration file was not found at: " + std::filesystem::absolute(configurationFileLocation).string());
    } else {
        config = ShapeBench::readConfiguration(configurationFile.value());
    }

    ShapeBench::BenchmarkConfiguration setup;
    uint32_t sampleDensity = config.at("commonExperimentSettings").at("pointSampleDensity");
    config.at("commonExperimentSettings").at("pointSampleDensity") = sampleDensity * 10;
    setup.configuration = config;


    // --- Compute and load dataset ---
    ShapeBench::Dataset dataset = ShapeBench::computeOrLoadCache(setup);

    double datasetSizeLimitGB = setup.configuration.at("datasetSettings").at("cacheSizeLimitGB");
    uint64_t datasetSizeLimitBytes = uint64_t(datasetSizeLimitGB * 1024.0 * 1024.0 * 1024.0);
    std::filesystem::path datasetCacheDirectory = setup.configuration.at("datasetSettings").at("compressedRootDir");
    std::string datasetDownloadBaseURL = setup.configuration.at("datasetSettings").at("objaverseDownloadBaseURL");
    bool verifyIntegrityOfDownloadedFiles = setup.configuration.at("datasetSettings").at("verifyFileIntegrity");
    ShapeBench::LocalDatasetCache* fileCache = new ShapeBench::LocalDatasetCache(datasetCacheDirectory, datasetDownloadBaseURL,  datasetSizeLimitBytes, verifyIntegrityOfDownloadedFiles);


    uint64_t randomSeed = config.at("randomSeed");
    ShapeBench::randomEngine engine(randomSeed);

    const uint32_t testVertexCount = 1000000;
    const uint32_t verticesPerMesh = 100;
    const float supportRadius = 0.38999998569488525;

    std::vector<ShapeBench::VertexInDataset> sampleVerticesSet = dataset.sampleVertices(engine(), testVertexCount, verticesPerMesh);
    std::vector<ShapeBench::TriangleMeshMICCIDescriptor> triangleDescriptors(testVertexCount);
    std::vector<ShapeBench::PointCloudMICCIDescriptor> pointCloudDescriptors(testVertexCount);


    fmt::println("Computing descriptors..");
    uint32_t meshCount = sampleVerticesSet.size() / verticesPerMesh;

    uint32_t completedCount = 0;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for(uint32_t meshIndex = 0; meshIndex < meshCount; meshIndex++) {
        uint32_t meshID = sampleVerticesSet.at(meshIndex * verticesPerMesh).meshID;
        const ShapeBench::DatasetEntry &entry = dataset.at(meshID);

        ShapeDescriptor::cpu::Mesh mesh = ShapeBench::readDatasetMesh(config, fileCache, entry);
        ShapeDescriptor::cpu::PointCloud pointCloud = ShapeBench::computePointCloud<ShapeBench::MICCIMethod_triangles>(mesh, config, 10, randomSeed);
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins(verticesPerMesh);
        std::vector<float> supportRadii(verticesPerMesh, supportRadius);

        for(uint32_t vertexBaseIndex = 0; vertexBaseIndex < verticesPerMesh; vertexBaseIndex++) {
            uint32_t vertexIndex = meshIndex * verticesPerMesh + vertexBaseIndex;
            ShapeBench::VertexInDataset vertexInDataset = sampleVerticesSet.at(vertexIndex);
            descriptorOrigins[vertexBaseIndex].vertex = mesh.vertices[vertexInDataset.vertexIndex];
            descriptorOrigins[vertexBaseIndex].normal = mesh.normals[vertexInDataset.vertexIndex];
        }

        ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> meshTriangleDescriptors
                = ShapeBench::micci::generateGrayscaleMICCIDescriptorMultiRadius(mesh, descriptorOrigins, supportRadii);

        ShapeDescriptor::cpu::array<ShapeBench::PointCloudMICCIDescriptor> meshPointCloudDescriptors
                = ShapeBench::micci::generateGrayscaleMICCIDescriptorsMultiRadius(pointCloud, descriptorOrigins, supportRadii);

        std::copy(meshTriangleDescriptors.content, meshTriangleDescriptors.content + verticesPerMesh, triangleDescriptors.data() + meshIndex * verticesPerMesh);
        std::copy(meshPointCloudDescriptors.content, meshPointCloudDescriptors.content + verticesPerMesh, pointCloudDescriptors.data() + meshIndex * verticesPerMesh);

        ShapeDescriptor::free(meshTriangleDescriptors);
        ShapeDescriptor::free(meshPointCloudDescriptors);
        ShapeDescriptor::free(pointCloud);
        ShapeDescriptor::free(mesh);
        ShapeDescriptor::free(descriptorOrigins);

        #pragma omp critical
        {
            completedCount++;
            fmt::print("    Completed {}/{} - ", completedCount, meshCount);
            ShapeBench::drawProgressBar(completedCount, meshCount);
            ShapeBench::printETA(startTime, completedCount, meshCount);
        }
    }


    fmt::println("Computing average descriptor distances..");
    float minThresholdLevel = config.at("methodSettings").at("MICCI-PointCloud").at("levelThresholdEstimation").at("minLevel");
    float maxThresholdLevel = config.at("methodSettings").at("MICCI-PointCloud").at("levelThresholdEstimation").at("maxLevel");
    uint32_t sampleCount = config.at("methodSettings").at("MICCI-PointCloud").at("levelThresholdEstimation").at("sampleCount");
    float thresholdLevelDelta = maxThresholdLevel - minThresholdLevel;

    startTime = std::chrono::steady_clock::now();



    std::ofstream distanceFile{"micci_threashold_distances.txt"};
    distanceFile << "threshold index, threshold, total distance" << std::endl;
    uint64_t lowestDistance = 0xFFFFFFFFFFFFFFFF;
    uint32_t lowestThresholdIndex = 0;

    for(uint32_t thresholdIndex = 0; thresholdIndex < sampleCount; thresholdIndex++) {
        float threshold = minThresholdLevel + (float(thresholdIndex) / float(sampleCount)) * thresholdLevelDelta;

        uint64_t totalDistance = 0;
        for(uint32_t descriptorIndex = 0; descriptorIndex < testVertexCount; descriptorIndex++) {
            const ShapeBench::PointCloudMICCIDescriptor& pointCloudDescriptor = pointCloudDescriptors.at(descriptorIndex);
            const ShapeBench::TriangleMeshMICCIDescriptor& groundTruthDescriptor = triangleDescriptors.at(descriptorIndex);
            ShapeBench::TriangleMeshMICCIDescriptor thresholdedDescriptor = ShapeBench::micci::discretiseMICCIImage(pointCloudDescriptor, threshold);
            uint64_t distanceToTriangleCounterpart = sumOfSquaredDifferences(thresholdedDescriptor, groundTruthDescriptor);
            totalDistance += distanceToTriangleCounterpart;
        }

        if(totalDistance < lowestDistance) {
            lowestDistance = totalDistance;
            lowestThresholdIndex = thresholdIndex;
        }

        distanceFile << fmt::format("{}, {}, {}", thresholdIndex, threshold, totalDistance) << std::endl;
        fmt::print("    Completed {}/{} - ", thresholdIndex+1, sampleCount);
        ShapeBench::drawProgressBar(thresholdIndex+1, sampleCount);
        ShapeBench::printETA(startTime, thresholdIndex+1, sampleCount);
    }

    float chosenThreshold = minThresholdLevel + (float(lowestThresholdIndex) / float(sampleCount)) * thresholdLevelDelta;
    fmt::println("Chose index {}, threshold {} as the optimal one.", lowestThresholdIndex, chosenThreshold);


    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> outImages(testVertexCount);
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> referenceImages(testVertexCount);

    for(uint32_t descriptorIndex = 0; descriptorIndex < testVertexCount; descriptorIndex++) {
        const ShapeBench::PointCloudMICCIDescriptor& pointCloudDescriptor = pointCloudDescriptors.at(descriptorIndex);
        const ShapeBench::TriangleMeshMICCIDescriptor& groundTruthDescriptor = triangleDescriptors.at(descriptorIndex);
        ShapeBench::TriangleMeshMICCIDescriptor thresholdedDescriptor = ShapeBench::micci::discretiseMICCIImage(pointCloudDescriptor, chosenThreshold);

        reinterpret_cast<ShapeBench::TriangleMeshMICCIDescriptor*>(outImages.content)[descriptorIndex] = thresholdedDescriptor;
        reinterpret_cast<ShapeBench::TriangleMeshMICCIDescriptor*>(referenceImages.content)[descriptorIndex] = groundTruthDescriptor;
    }

    ShapeDescriptor::writeDescriptorImages(outImages, "micci_images_thresholded.png", false);
    ShapeDescriptor::writeDescriptorImages(referenceImages, "micci_images_groundtruth.png", false);

    fmt::println("Complete.");

    return 0;
}
