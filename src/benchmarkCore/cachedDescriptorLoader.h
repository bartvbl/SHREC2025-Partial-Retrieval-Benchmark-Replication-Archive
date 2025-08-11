#pragma once

#include <vector>
#include <random>
#include "dataset/Dataset.h"
#include "replication/RandomSubset.h"
#include "randomEngine.h"
#include "benchmarkCore/common-procedures/meshLoader.h"
#include "utils/prettyprint.h"
#include "benchmarkCore/common-procedures/descriptorGenerator.h"

namespace ShapeBench {

    template<typename T>
    inline T& getRef(T* p) { return *p; }

    template<typename DescriptorMethod, typename DescriptorType>
    std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> computeReferenceDescriptors(const std::vector<ShapeBench::VertexInDataset>& representativeSet,
                                                                                                     const nlohmann::json& config,
                                                                                                     const ShapeBench::Dataset& dataset,
                                                                                                     ShapeBench::LocalDatasetCache* fileCache,
                                                                                                     uint64_t randomSeed,float supportRadius,
                                                                                                     ShapeBench::RandomSubset* randomSubsetToReplicate = nullptr) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        ShapeBench::randomEngine randomEngine(randomSeed);
        std::vector<uint64_t> randomSeeds(representativeSet.size());
        for(uint32_t i = 0; i < representativeSet.size(); i++) {
            randomSeeds.at(i) = randomEngine();
        }

        std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> representativeDescriptors(representativeSet.size());
        uint32_t completedCount = 0;

        #pragma omp parallel for schedule(dynamic) default(none) shared(representativeSet, randomSubsetToReplicate, supportRadius, fileCache, dataset, config, randomSeeds, representativeDescriptors, completedCount, std::cout)
        for(int i = 0; i < representativeSet.size(); i++) {
            uint32_t currentMeshID = representativeSet.at(i).meshID;
            if(i > 0 && representativeSet.at(i - 1).meshID == currentMeshID) {
                continue;
            }

            bool shouldReplicate = randomSubsetToReplicate != nullptr;
            bool sequenceContainsItemToReplicate = randomSubsetToReplicate != nullptr && randomSubsetToReplicate->contains(i);

            uint32_t sameMeshCount = 1;
            for(int j = i + 1; j < representativeSet.size(); j++) {
                uint32_t meshID = representativeSet.at(j).meshID;
                if(shouldReplicate && randomSubsetToReplicate->contains(j)) {
                    sequenceContainsItemToReplicate = true;
                }
                if(currentMeshID != meshID) {
                    break;
                }
                sameMeshCount++;
            }

            if(shouldReplicate && !sequenceContainsItemToReplicate) {
                continue;
            }

            std::vector<DescriptorType> outputDescriptors(sameMeshCount);
            std::vector<ShapeDescriptor::OrientedPoint> descriptorOrigins(sameMeshCount);
            std::vector<float> radii(sameMeshCount, supportRadius);

            const ShapeBench::DatasetEntry& entry = dataset.at(representativeSet.at(i).meshID);
            ShapeDescriptor::cpu::Mesh mesh = ShapeBench::readDatasetMesh(config, fileCache, entry);

            for(int j = 0; j < sameMeshCount; j++) {
                uint32_t entryIndex = j + i;
                ShapeBench::VertexInDataset vertex = representativeSet.at(entryIndex);
                descriptorOrigins.at(j).vertex = mesh.vertices[vertex.vertexIndex];
                descriptorOrigins.at(j).normal = mesh.normals[vertex.vertexIndex];
            }

            std::vector<bool> allDescriptorsActive(sameMeshCount, true);

            ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> originArray(descriptorOrigins.size(), descriptorOrigins.data());

            uint64_t runSeed = randomSeeds.at(i);
            bool successfulRun = false;
            uint32_t failedRunCount = 0;
            while(!successfulRun) {
                try {
                ShapeBench::computeDescriptors<DescriptorMethod, DescriptorType>(mesh, originArray, allDescriptorsActive, config, radii, runSeed, runSeed, 1.0f, outputDescriptors);
                successfulRun = true;
                } catch(const std::exception& e) {
                    std::cout << "Failed to generate descriptors while computing cached descriptor set" << std::endl;
                    std::cout << "Reason: " << e.what() << std::endl;
                    std::cout << "Trying a different seed instead" << std::endl;
                    std::cout << "Mesh index: " << i << std::endl;
                    std::minstd_rand0 tempEngine(runSeed);
                    runSeed = tempEngine();
                    failedRunCount++;
                    if(failedRunCount == 10) {
                        /*ShapeDescriptor::cpu::PointCloud pointCloud = computePointCloud<DescriptorMethod>(mesh, config, 1, runSeed);
                        DebugRenderer renderer;
                        renderer.drawMesh(mesh);
                        ShapeDescriptor::writeOBJ(mesh, "weirdmesh.obj");
                        renderer.drawPointCloud(pointCloud, {0, 0, 1});
                        std::cout << "Number of descriptors: " << originArray.length << std::endl;
                        for(int i = 0; i < originArray.length; i++) {
                            std::cout << originArray.content[i].vertex << std::endl;
                            float size = 0.05;
                            renderer.drawLine(originArray.content[i].vertex + ShapeDescriptor::cpu::float3{0, 0, -size}, originArray.content[i].vertex + ShapeDescriptor::cpu::float3{0, 0, size}, allDescriptorsActive.at(i) ? ShapeDescriptor::cpu::float3{0, 1, 0} : ShapeDescriptor::cpu::float3{1, 0, 0});
                            renderer.drawLine(originArray.content[i].vertex + ShapeDescriptor::cpu::float3{0, -size, 0}, originArray.content[i].vertex + ShapeDescriptor::cpu::float3{0, size, 0}, allDescriptorsActive.at(i) ? ShapeDescriptor::cpu::float3{0, 1, 0} : ShapeDescriptor::cpu::float3{1, 0, 0});
                            renderer.drawLine(originArray.content[i].vertex + ShapeDescriptor::cpu::float3{-size, 0, 0}, originArray.content[i].vertex + ShapeDescriptor::cpu::float3{size, 0, 0}, allDescriptorsActive.at(i) ? ShapeDescriptor::cpu::float3{0, 1, 0} : ShapeDescriptor::cpu::float3{1, 0, 0});
                        }
                        renderer.show("Testing");
                        renderer.waitForClose();
                        ShapeDescriptor::free(pointCloud);*/

                        throw std::runtime_error("Failed descriptor generation too many times! Mesh file: " + entry.meshFile.string());
                    }
                }
            }
            
            for(int j = 0; j < sameMeshCount; j++) {
                uint32_t entryIndex = j + i;
                ShapeBench::VertexInDataset vertex = representativeSet.at(entryIndex);
                representativeDescriptors.at(entryIndex).descriptor = outputDescriptors.at(j);
                representativeDescriptors.at(entryIndex).meshID = vertex.meshID;
                representativeDescriptors.at(entryIndex).vertexIndex = vertex.vertexIndex;
                representativeDescriptors.at(entryIndex).vertex = descriptorOrigins.at(j);
            }

            ShapeDescriptor::free(mesh);

    #pragma omp atomic
            completedCount += sameMeshCount;

            if(completedCount % 100 == 0 || completedCount == representativeSet.size()) {
                std::cout << "\r    ";
                ShapeBench::drawProgressBar(completedCount, representativeSet.size());
                std::cout << " " << completedCount << "/" << representativeSet.size() << std::flush;
                malloc_trim(0);
            }
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "    Complete." << std::endl;
        std::cout << "    Elapsed time: ";
        std::cout << ShapeBench::durationToString(end - begin);
        std::cout << std::endl;

        return representativeDescriptors;
    }



    template<typename DescriptorType, typename DescriptorMethod, typename inputRepresentativeSetType>
    std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> computeDescriptorsOrLoadCached(
            const nlohmann::json &configuration,
            const ShapeBench::Dataset &dataset,
            ShapeBench::LocalDatasetCache* fileCache,
            float supportRadius,
            uint64_t representativeSetRandomSeed,
            const std::vector<inputRepresentativeSetType> &representativeSet,
            std::string name) {
        const std::filesystem::path descriptorCacheFile = std::filesystem::path(std::string(configuration.at("cacheDirectory"))) / (name + "Descriptors-" + DescriptorMethod::getName() + ".dat");
        std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> referenceDescriptors;

        bool replicateSubset = configuration.at("replicationOverrides").at(name + "DescriptorSet").at("recomputeRandomSubset");
        uint32_t randomSubsetSize = configuration.at("replicationOverrides").at(name + "DescriptorSet").at("randomSubsetSize");
        bool replicateEntirely = configuration.at("replicationOverrides").at(name + "DescriptorSet").at("recomputeEntirely");

        if(!std::filesystem::exists(descriptorCacheFile)) {
            std::cout << "    No cached " + name + " descriptors were found." << std::endl;
            std::cout << "    Computing " + name + " descriptors.." << std::endl;
            referenceDescriptors = computeReferenceDescriptors<DescriptorMethod, DescriptorType>(representativeSet, configuration, dataset, fileCache, representativeSetRandomSeed, supportRadius);
            std::cout << "    Finished computing " + name + " descriptors. Writing archive file.." << std::endl;
            ShapeDescriptor::writeCompressedDescriptors<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>(descriptorCacheFile, {referenceDescriptors.size(), referenceDescriptors.data()});

            std::cout << "    Checking integrity of written data.." << std::endl;
            std::cout << "        Reading written file.." << std::endl;
            ShapeDescriptor::cpu::array<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> readDescriptors = ShapeDescriptor::readCompressedDescriptors<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>(descriptorCacheFile, 8);
            assert(referenceDescriptors.size() == readDescriptors.length);
            for(uint32_t i = 0; i < referenceDescriptors.size(); i++) {
                char* basePointerA = reinterpret_cast<char*>(&referenceDescriptors.at(i));
                char* basePointerB = reinterpret_cast<char*>(&readDescriptors.content[i]);
                for(uint32_t j = 0; j < sizeof(DescriptorType); j++) {
                    if(basePointerA[j] != basePointerB[j]) {
                        throw std::runtime_error("Descriptors at index " + std::to_string(i) + " are not identical!");
                    }
                }
            }
            std::cout << "    Check complete, no errors detected." << std::endl;
            ShapeDescriptor::free(readDescriptors);
        } else {
            std::cout << "    Loading cached " + name + " descriptors.." << std::endl;
            ShapeDescriptor::cpu::array<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> readDescriptors = ShapeDescriptor::readCompressedDescriptors<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>(descriptorCacheFile, 8);
            referenceDescriptors.resize(readDescriptors.length);
            std::copy(readDescriptors.content, readDescriptors.content + readDescriptors.length, referenceDescriptors.begin());
            ShapeDescriptor::free(readDescriptors);
            std::cout << "    Successfully loaded " << referenceDescriptors.size() << " descriptors" << std::endl;
            std::string errorMessage = "Mismatch detected between the cached number of descriptors ("
                                       + std::to_string(referenceDescriptors.size()) + ") and the requested number of descriptors ("
                                       + std::to_string(representativeSet.size()) + ").";
            if(referenceDescriptors.size() < representativeSet.size()) {
                throw std::runtime_error("ERROR: " + errorMessage + " The number of cached descriptors is not sufficient. "
                                                                    "Consider regenerating the cache by deleting the file: " + descriptorCacheFile.string());
            } else if(referenceDescriptors.size() > representativeSet.size()) {
                std::cout << "    WARNING: " + errorMessage + " Since the cache contains more descriptors than necessary, execution will continue." << std::endl;
            }

            if(replicateSubset || replicateEntirely) {
                if(replicateSubset && replicateEntirely) {
                    std::cout << "    NOTE: replication of a random subset *and* the entire descriptor set was requested. The entire set will be replicated." << std::endl;
                }
                std::cout << "    Replication of " << name << " descriptor set has been enabled. Performing replication.." << std::endl;
                uint32_t numberOfDescriptorsToReplicate = replicateEntirely ? readDescriptors.length : std::min<uint32_t>(readDescriptors.length, randomSubsetSize);
                uint64_t replicationRandomSeed = configuration.at("replicationOverrides").at("replicationRandomSeed");
                ShapeBench::RandomSubset subset(0, readDescriptors.length, numberOfDescriptorsToReplicate, replicationRandomSeed);
                std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> replicatedDescriptors;
                std::cout << "    Computing " << numberOfDescriptorsToReplicate << " descriptors.." << std::endl;
                replicatedDescriptors = computeReferenceDescriptors<DescriptorMethod, DescriptorType>(representativeSet, configuration, dataset, fileCache, representativeSetRandomSeed, supportRadius, &subset);
                std::cout << "    Comparing computed descriptors against those in the cache.." << std::endl;
                uint32_t inconsistentDescriptorCount = 0;
                uint32_t replicatedDescriptorCount = 0;
                for(uint32_t descriptorIndex = 0; descriptorIndex < representativeSet.size(); descriptorIndex++) {
                    if(!subset.contains(descriptorIndex)) {
                        continue;
                    }
                    replicatedDescriptorCount++;
                    uint8_t* descriptorA = reinterpret_cast<uint8_t*>(&replicatedDescriptors.at(descriptorIndex).descriptor);
                    uint8_t* descriptorB = reinterpret_cast<uint8_t*>(&referenceDescriptors.at(descriptorIndex).descriptor);
                    for(uint32_t byteIndex = 0; byteIndex < sizeof(DescriptorType); byteIndex++) {
                        if(descriptorA[byteIndex] != descriptorB[byteIndex]) {
                            inconsistentDescriptorCount++;
                            //throw std::logic_error("FATAL: Descriptors at index " + std::to_string(descriptorIndex) + " failed to replicate at byte " + std::to_string(byteIndex) + "!");
                            continue;
                        }
                    }
                }
                std::cout << "    Comparison complete. Number of identical descriptors: " << (replicatedDescriptorCount - inconsistentDescriptorCount) << " / " << replicatedDescriptorCount << std::endl;
                if(inconsistentDescriptorCount > 0) {
                    std::cout << "    Some inconsistent descriptors were detected. This can happen due to rounding errors, so it's not necessarily a problem." << std::endl;
                    std::cout << "    Would you like to continue anyway? yes/no: ";
                    std::string answer;
                    std::cin >> answer;
                    if(answer.at(0) != 'y' && answer.at(0) != 'Y') {
                        throw std::logic_error("Aborted.");
                    }
                }
            }
        }

        if(!referenceDescriptors.empty()) {
            const bool scanReadDescriptorsForNans = true;
            constexpr bool descriptorStoresFloats = std::is_floating_point_v<std::remove_reference_t<decltype(getRef(referenceDescriptors.at(0).descriptor.contents))>>;
            if(!descriptorStoresFloats) {
                std::cout << "    No NaN sanity check was performed on descriptors: descriptor contents are not floating point numbers." << std::endl;
            }
            if(scanReadDescriptorsForNans && descriptorStoresFloats) {
                uint64_t totalNaNCount = 0;
                for(uint32_t i = 0; i < referenceDescriptors.size(); i++) {
                    const DescriptorType& descriptor = referenceDescriptors.at(i).descriptor;
                    const uint32_t floatCount = sizeof(DescriptorType) / sizeof(std::remove_reference_t<decltype(descriptor.contents)>);
                    for(int j = 0; j < floatCount; j++) {
                        if(std::isnan(descriptor.contents[j])) {
                            totalNaNCount++;
                        }
                    }
                }
                if(totalNaNCount > 0) {
                    throw std::runtime_error(fmt::format("A total of {} NaN values were detected in the cached reference set. This yields invalid benchmark results. Please recompute the reference set, and instrument your code to detect NaN errors. These can occur due to bugs or edge cases in input data.", totalNaNCount));
                }
                std::cout << "    No NaN values were detected in the cached reference descriptor set." << std::endl;
            }
        }

        return referenceDescriptors;
    }
}

