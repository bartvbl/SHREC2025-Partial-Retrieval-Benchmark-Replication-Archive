#pragma once

#include "nlohmann/json.hpp"
#include "dataset/Dataset.h"
#include "benchmarkCore/randomEngine.h"
#include "utils/prettyprint.h"
#include "benchmarkCore/common-procedures/meshLoader.h"
#include "benchmarkCore/common-procedures/descriptorGenerator.h"
#include "utils/meshPrimitiveGenerator.h"
#include "syntheticMeshGenerator.h"


namespace ShapeBench {
    constexpr float noSampleCountScaleFactor = 1.0;
    constexpr bool enforceSampleCountLowerBound = false;
    constexpr bool enforceSampleCountUpperBound = false;



    template<typename DescriptorMethod>
    double createRandomObjectScene(const nlohmann::json &config, const Dataset &dataset, LocalDatasetCache *fileCache,
                                 const int objectsPerSample, const uint32_t maxPointSampleCount,
                                 randomEngine &variableInputSizeEngine, ShapeDescriptor::cpu::Mesh &sceneMesh) {
        double sceneMeshSurfaceArea = 0;
        bool validSceneCreated = false;
        while(!validSceneCreated) {
            if(sceneMesh.vertexCount > 0) {
                ShapeDescriptor::free(sceneMesh);
            }

            validSceneCreated = true;
            std::vector<VertexInDataset> objectsToLoad = dataset.sampleVertices(variableInputSizeEngine(), objectsPerSample, 1);

            uint64_t totalVertexCount = 0;
            for (const VertexInDataset &vertex: objectsToLoad) {
                totalVertexCount += dataset.at(vertex.meshID).vertexCount;
            }
            sceneMesh = ShapeDescriptor::cpu::Mesh(totalVertexCount);

            uint64_t nextVertexIndex = 0;

            for (int j = 0; j < objectsToLoad.size(); j++) {
                ShapeDescriptor::cpu::Mesh loadedMesh = readDatasetMesh(config, fileCache, dataset.at(objectsToLoad.at(0).meshID));
                if (DescriptorMethod::usesPointCloudInput()) {
                    double meshArea = ShapeDescriptor::calculateMeshSurfaceArea(loadedMesh);

                    uint32_t sampleCount = computeSampleCount(sceneMeshSurfaceArea + meshArea,
                                                              noSampleCountScaleFactor, config,
                                                              enforceSampleCountLowerBound);
                    if (sampleCount >= maxPointSampleCount) {
                        // It's important for the execution time measurements that the surface density is consistent
                        // to allow comparison between iterations. We therefore don't build any scenes with meshes that
                        // would exceed the limit of the benchmark
                        ShapeDescriptor::free(loadedMesh);
                        continue;
                    }
                    sceneMeshSurfaceArea += meshArea;
                }

                std::copy(loadedMesh.vertices, loadedMesh.vertices + loadedMesh.vertexCount,
                          sceneMesh.vertices + nextVertexIndex);
                std::copy(loadedMesh.normals, loadedMesh.normals + loadedMesh.vertexCount,
                          sceneMesh.normals + nextVertexIndex);
                if (loadedMesh.vertexColours != nullptr) {
                    std::copy(loadedMesh.vertexColours, loadedMesh.vertexColours + loadedMesh.vertexCount,
                              sceneMesh.vertexColours + nextVertexIndex);
                }

                nextVertexIndex += loadedMesh.vertexCount;
                ShapeDescriptor::free(loadedMesh);
            }

            if (DescriptorMethod::usesPointCloudInput() && sceneMeshSurfaceArea == 0) {
                // Degenerate mesh
                ShapeDescriptor::free(sceneMesh);
                validSceneCreated = false;
            } else if(sceneMesh.vertexCount == 0) {
                ShapeDescriptor::free(sceneMesh);
                validSceneCreated = false;
            }
        }
        return sceneMeshSurfaceArea;
    }


    template<typename DescriptorType, typename DescriptorMethod>
    nlohmann::json runSyntheticExecutionTimeExperiment(const nlohmann::json &generalConfig,
                                             const nlohmann::json &experimentConfig,
                                             uint64_t randomSeed,
                                             const ShapeDescriptor::cpu::Mesh& unitCubeMesh,
                                             const ShapeDescriptor::cpu::Mesh& unitSphereMesh) {

        // Reading out all configuration parameters for the experiment
        const uint32_t syntheticExperimentsDescriptorCount = experimentConfig.at("descriptorsPerMesh");
        const float originRange = experimentConfig.at("randomOriginRange");
        const float syntheticExperimentsSupportRadius = experimentConfig.at("supportRadius");
        const AllowedLocation insideOrOutsideOfSupportVolume = experimentConfig.at("allowedLocation") == "inside" ? AllowedLocation::INSIDE : AllowedLocation::OUTSIDE;
        const uint32_t instanceCountMin = experimentConfig.at("meshInstanceCountRange")[0];
        const uint32_t instanceCountStep = experimentConfig.at("meshInstanceCountStep");
        const uint32_t instanceCountMax = experimentConfig.at("meshInstanceCountRange")[1];
        const uint32_t instancesPerLocation = experimentConfig.at("meshInstanceCountPerChosenPosition");
        const float cubeScaleFactor = experimentConfig.at("meshInstanceScaleFactor");
        const float pointCloudDensityScaleFactor = experimentConfig.at("pointCloudDensityScaleFactor");
        const bool generateSampleMesh = experimentConfig.at("generateSampleMeshes");
        const std::filesystem::path sampleMeshOutputDirectory = std::string(experimentConfig.at("generatedMeshDirectory"));
        const uint32_t repetitionCount = experimentConfig.at("repetitionCount");
        const std::string meshInstanceToUse = experimentConfig.at("mesh");
        const ShapeDescriptor::cpu::Mesh& meshInstanceToPlace = meshInstanceToUse == "cube" ? unitCubeMesh
                                                              : meshInstanceToUse == "sphere" ? unitSphereMesh
                                                              : throw std::runtime_error(fmt::format("Unknown mesh instance specified: {}", meshInstanceToUse));

        ShapeBench::randomEngine experimentEngine(randomSeed);
        ShapeBench::randomEngine pointCloudSamplingEngine(randomSeed);

        nlohmann::json executionTimeMeasurements;

        double instanceMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(meshInstanceToPlace);
        double scaledInstanceMeshArea = cubeScaleFactor * cubeScaleFactor * instanceMeshArea;


        std::uniform_real_distribution<float> originRangeDistribution(-originRange, originRange);
        std::vector<DescriptorType> outputDescriptors(syntheticExperimentsDescriptorCount);

        for(uint32_t instanceCount = instanceCountMin; instanceCount <= instanceCountMax; instanceCount += instanceCountStep) {
            ShapeDescriptor::cpu::float3 meshOrigin = {originRangeDistribution(experimentEngine), originRangeDistribution(experimentEngine), originRangeDistribution(experimentEngine)};
            ShapeDescriptor::cpu::float2 meshOrientation = ShapeBench::generateRandomSphericalCoordinates(experimentEngine);
            ShapeDescriptor::cpu::float3 meshOrientationDirection = ShapeBench::directionVectorFromSphericalCoordinates(meshOrientation);
            ShapeDescriptor::cpu::Mesh generatedMesh = generateSyntheticMesh<DescriptorMethod>(meshInstanceToPlace, instanceCount, instancesPerLocation, cubeScaleFactor, meshOrigin, meshOrientation, syntheticExperimentsSupportRadius, insideOrOutsideOfSupportVolume, experimentEngine);

            if(generateSampleMesh) {
                fmt::println("Writing mesh..");
                std::filesystem::create_directories(sampleMeshOutputDirectory);
                std::string filename = fmt::format("{}-{}-{}-{}.obj", std::string(experimentConfig.at("allowedLocation")),
                                                                   std::string(experimentConfig.at("mesh")),
                                                                   instanceCount, instancesPerLocation);
                std::filesystem::path meshFilePath = sampleMeshOutputDirectory / filename;
                ShapeDescriptor::writeOBJ(generatedMesh, meshFilePath);
            }

            double generatedMeshArea = double(instanceCount) * scaledInstanceMeshArea;

            ShapeDescriptor::cpu::PointCloud pointCloud;
            if (DescriptorMethod::usesPointCloudInput()) {
                size_t pointCloudSamplingSeed = pointCloudSamplingEngine();
                uint32_t sampleCount = computeSampleCount(generatedMeshArea, noSampleCountScaleFactor, generalConfig, enforceSampleCountLowerBound, enforceSampleCountUpperBound);
                uint32_t scaledSampleCount = uint32_t(pointCloudDensityScaleFactor * float(sampleCount));
                pointCloud = computePointCloud<DescriptorMethod>(generatedMesh, generalConfig, noSampleCountScaleFactor, pointCloudSamplingSeed, scaledSampleCount, enforceSampleCountLowerBound);
            }

            ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins(syntheticExperimentsDescriptorCount);
            for (int originIndex = 0; originIndex < syntheticExperimentsDescriptorCount; originIndex++) {
                descriptorOrigins.content[originIndex].vertex = meshOrigin;
                descriptorOrigins.content[originIndex].normal = meshOrientationDirection;
            }

            std::vector<float> supportRadii(descriptorOrigins.length, syntheticExperimentsSupportRadius);
            std::vector<bool> enabledMasks(descriptorOrigins.length, true);
            std::vector<double> executionTimes(repetitionCount);

            fmt::println("Generated scene: {} - {} - {}\n    ", instanceCount, generatedMesh.vertexCount/3, pointCloud.pointCount);

            for(uint32_t repetition = 0; repetition < repetitionCount; repetition++) {
                std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();
                try {
                    computeDescriptors<DescriptorMethod, DescriptorType>(generatedMesh, pointCloud, descriptorOrigins,
                                                                         enabledMasks, generalConfig, supportRadii,
                                                                         experimentEngine(), outputDescriptors);
                } catch(const std::exception& e) {}
                std::chrono::time_point<std::chrono::steady_clock> endTime = std::chrono::steady_clock::now();

                uint64_t timeTakenNanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
                double timeTakenSeconds = double(timeTakenNanoseconds) / 1000000000.0;
                executionTimes.at(repetition) = timeTakenSeconds;
                fmt::print("{}, ", timeTakenSeconds);
            }
            fmt::println("\n");

            uint64_t workloadSize = 0;

            std::string workloadName = DescriptorMethod::usesPointCloudInput() ? "point" : "triangle";
            if(DescriptorMethod::usesPointCloudInput()) {
                workloadSize = pointCloud.pointCount;
            } else {
                workloadSize = generatedMesh.vertexCount / 3;
            }

            // std::cout << descriptorOrigins.length << ", " << workloadSize << ", " << descriptorOrigins.length*workloadSize << ", " << timeTakenSeconds << ", " << ((descriptorOrigins.length*workloadSize)/timeTakenSeconds) << std::endl;

            nlohmann::json resultsForInstanceCount = {};
            resultsForInstanceCount["descriptorCount"] = descriptorOrigins.length;
            resultsForInstanceCount["workloadPerDescriptor"] = workloadSize;
            resultsForInstanceCount["workloadItemsProcessed"] = descriptorOrigins.length*workloadSize;
            resultsForInstanceCount["executionTimes"] = executionTimes;
            executionTimeMeasurements.push_back(resultsForInstanceCount);

            if(DescriptorMethod::usesPointCloudInput()) {
                ShapeDescriptor::free(pointCloud);
            }

            ShapeDescriptor::free(generatedMesh);
        }

        return executionTimeMeasurements;
    }

    template<typename DescriptorMethod, typename DescriptorType>
    void computeDescriptorMethodExecutionTime(const nlohmann::json& config, const Dataset& dataset, ShapeBench::LocalDatasetCache* fileCache, uint64_t randomSeed, float supportRadius,
                                              const std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>& referenceDescriptors) {

        nlohmann::json executionTimeResults;
        executionTimeResults["results"]["generation"]["syntheticScene"] = {};
        executionTimeResults["results"]["generation"]["realScene"] = {};
        executionTimeResults["configuration"] = config;
        executionTimeResults["workloadType"] = DescriptorMethod::usesPointCloudInput() ? "point" : "triangle";

        const nlohmann::json& executionTimeSettings = config.at("executionTimeMeasurement");
        const nlohmann::json& experimentsToRun = executionTimeSettings.at("syntheticExperiments");
        const nlohmann::json& comparisonTimeSettings = executionTimeSettings.at("descriptorComparisonSettings");
        nlohmann::json commonExperimentSettings = executionTimeSettings.at("syntheticExperimentsSharedSettings");


        fmt::println(" --- comparison time experiments ---");


        std::vector<DescriptorType> consolidatedDescriptors;
        consolidatedDescriptors.reserve(referenceDescriptors.size());
        uint32_t comparisonTimeRepetitionCount = comparisonTimeSettings.at("iterationCount");
        // Filter out all memory that's not related to the descriptor itself. Should keep this better aligned in memory.
        for(const ShapeBench::DescriptorOfVertexInDataset<DescriptorType>& descriptor : referenceDescriptors) {
            consolidatedDescriptors.push_back(descriptor.descriptor);
        }

        // We have two indices chasing each other. They just need to be far enough apart so that the other descriptor is not in cache at the same time
        uint32_t descriptorIndexA = 0;
        uint32_t descriptorIndexB = referenceDescriptors.size() / 2;
        float totalDistance = 0;
        uint64_t comparisonsDone = comparisonTimeRepetitionCount * referenceDescriptors.size();


        std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();

        for(uint64_t repetition = 0; repetition < comparisonsDone; repetition++) {
            // Doing something with the result to ensure it does not get optimised away
            totalDistance += DescriptorMethod::computeDescriptorDistance(consolidatedDescriptors.at(descriptorIndexA),
                                                        consolidatedDescriptors.at(descriptorIndexB), std::numeric_limits<float>::max());

            descriptorIndexA++;
            descriptorIndexB++;
            descriptorIndexA %= referenceDescriptors.size();
            descriptorIndexB %= referenceDescriptors.size();
        }

        std::chrono::time_point<std::chrono::steady_clock> endTime = std::chrono::steady_clock::now();

        uint64_t timeTakenNanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
        double timeTakenSeconds = double(timeTakenNanoseconds) / 1000000000.0;
        double comparisonsPerSecond = double(comparisonsDone) / timeTakenSeconds;
        executionTimeResults["results"]["comparison"]["comparisonsPerSecond"] = comparisonsPerSecond;
        fmt::println("    Comparisons per second: {}", comparisonsPerSecond);
        fmt::println("    Total time seconds: {}", timeTakenSeconds);
        fmt::println("    Comparisons done: {}", comparisonsDone);
        fmt::println("    Total distance: {}", totalDistance);



        fmt::println(" --- synthetic scene experiments ---");

        ShapeDescriptor::cpu::Mesh unitCubeMesh = ShapeBench::generateUnitCubeMesh();
        ShapeDescriptor::cpu::Mesh unitSphereMesh = ShapeBench::generateUnitSphereMesh(commonExperimentSettings.at("sphereInstanceLayerCount"),
                                                                                       commonExperimentSettings.at("sphereInstanceSliceCount"));




        for(uint32_t experimentIndex = 0; experimentIndex < experimentsToRun.size(); experimentIndex++) {
            nlohmann::json experimentConfig = experimentsToRun.at(experimentIndex);
            fmt::println("    Experiment {}/{}: {}", experimentIndex+1, experimentsToRun.size(), std::string(experimentConfig.at("description")));
            commonExperimentSettings = executionTimeSettings.at("syntheticExperimentsSharedSettings");
            commonExperimentSettings.merge_patch(experimentConfig);

            nlohmann::json measurements = runSyntheticExecutionTimeExperiment<DescriptorType, DescriptorMethod>(config, commonExperimentSettings, randomSeed, unitCubeMesh, unitSphereMesh);
            executionTimeResults["results"]["generation"]["syntheticScene"].push_back(measurements);
        }





        std::string unique = ShapeDescriptor::generateUniqueFilenameString();
        std::filesystem::path outputDirectory = std::filesystem::path(std::string(config.at("resultsDirectory"))) / "execution_times";
        if(!std::filesystem::exists(outputDirectory)) {
            std::filesystem::create_directories(outputDirectory);
        }
        std::filesystem::path outputPath = outputDirectory / fmt::format("synthetic_{}_{}.json", DescriptorMethod::getName(), unique);
        std::ofstream outputFile(outputPath);
        outputFile << executionTimeResults.dump(4);



        ShapeDescriptor::free(unitCubeMesh);
        ShapeDescriptor::free(unitSphereMesh);




/*
        for(uint32_t i = 0; i < variableInputSizeCount; i++) {
            ShapeDescriptor::cpu::Mesh sceneMesh;




            double sceneMeshSurfaceArea = createRandomObjectScene<DescriptorMethod>(config, dataset, fileCache, 1,
                                                         maxPointSampleCount, engine_experiment1, sceneMesh);


            ShapeDescriptor::cpu::PointCloud pointCloud;
            if (DescriptorMethod::usesPointCloudInput()) {
                size_t pointCloudSamplingSeed = engine_pointCloudsSampling_experiment1();

                uint32_t sampleCount = computeSampleCount(sceneMeshSurfaceArea, noSampleCountScaleFactor, config, enforceSampleCountLowerBound);

                pointCloud = computePointCloud<DescriptorMethod>(sceneMesh, config, noSampleCountScaleFactor, pointCloudSamplingSeed, sampleCount, enforceSampleCountLowerBound);
            }

            ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins(variableInputDescriptorCount);
            std::uniform_int_distribution<uint32_t> randomVertexDistribution(0, sceneMesh.vertexCount);


            uint32_t threadCount = 1;

            const uint32_t stepCount = 10;
            const uint32_t descriptorsPerStep = variableInputDescriptorCount / stepCount;
            std::array<double, stepCount> batchTimes;
            batchTimes.fill(0);

            std::vector<DescriptorType> outputDescriptors(variableInputDescriptorCount);

            for (int originIndex = 0; originIndex < variableInputDescriptorCount; originIndex++) {
                uint32_t randomVertexIndex = randomVertexDistribution(descriptorOriginsEngine);
                descriptorOrigins.content[originIndex].vertex = sceneMesh.vertices[randomVertexIndex];
                descriptorOrigins.content[originIndex].normal = sceneMesh.normals[randomVertexIndex];
            }

            uint64_t workloadSize = 0;

            std::string workloadName = DescriptorMethod::usesPointCloudInput() ? "point" : "triangle";
            if(DescriptorMethod::usesPointCloudInput()) {
                workloadSize = pointCloud.pointCount;
            } else {
                workloadSize = sceneMesh.vertexCount / 3;
            }

            // Threads and batch sizes multiply the amount of work done
            workloadSize *= threadCount;

            std::cout << "\r        Processing " << (i+1) << "/" << variableInputSizeCount << ", " << workloadName << " base count, " << workloadSize;

            for(int radius = 0; radius <= 10; radius++) {
                float supportRadiusToUse = float(radius) * 0.1;//(supportRadius / 10.0);
                for (int32_t step = stepCount - 1; step >= 0; step--) {
                    if(step != stepCount - 1) {
                        continue;
                    }
                    uint32_t descriptorsToCompute = (step + 1) * descriptorsPerStep;

                    std::vector<float> supportRadii(descriptorsToCompute, supportRadiusToUse);

                    descriptorOrigins.length = descriptorsToCompute;

                    std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();

                    // Trick to disable OpenMP threading, because OpenMP does not spawn additional threads within parallel sections
                    #pragma omp parallel
                    {
                        if (omp_get_thread_num() == 0) {
                            ShapeBench::computeDescriptors<DescriptorMethod, DescriptorType>(sceneMesh, pointCloud,
                                                                                             descriptorOrigins, config,
                                                                                             supportRadii,
                                                                                             engine_experiment1(),
                                                                                             outputDescriptors);
                        }
                    };

                    std::chrono::time_point<std::chrono::steady_clock> endTime = std::chrono::steady_clock::now();

                    uint64_t timeTakenMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                            endTime - startTime).count();
                    double timeTakenSeconds = double(timeTakenMilliseconds) / 1000.0;
                    if (step == stepCount - 1) {
                        std::cout << ", " << timeTakenSeconds << ", " << ((descriptorsToCompute*workloadSize)/timeTakenSeconds) << std::flush;
                    }

                    batchTimes.at(step) = timeTakenSeconds;
                }
            }

            if(DescriptorMethod::usesPointCloudInput()) {
                ShapeDescriptor::free(pointCloud);
            }

            ShapeDescriptor::free(sceneMesh);






            std::array<double, stepCount> workloadSizes;
            for(uint32_t step = 0; step < stepCount; step++) {
                workloadSizes.at(step) = workloadSize * (step + 1) * descriptorsPerStep;
            }




            std::cout << ", overhead time (s), throughput (" << workloadName << "s/s): ";

            std::cout << std::endl << "        ";
            ShapeBench::drawProgressBar(i, variableInputSizeCount);
            std::cout << std::flush;

        }*/

        std::cout << "Execution time measurement completed." << std::endl;
    }
}
