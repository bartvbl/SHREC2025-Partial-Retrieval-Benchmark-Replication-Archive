#include "ResultDumper.h"
#include "benchmarkCore/BenchmarkConfiguration.h"
#include <git.h>
#include <cuda_runtime_api.h>

nlohmann::json getGPUInfo() {
    nlohmann::json deviceInfo = {};
    int32_t deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0) {
        return deviceInfo;
    }


    cudaDeviceProp device_information;
    int32_t deviceID;
    cudaGetDevice(&deviceID);

    cudaError_t errorCode = cudaGetDeviceProperties(&device_information, deviceID);
    if(errorCode == cudaErrorInvalidDevice) {
        return deviceInfo;
    }
    deviceInfo["gpuName"] = std::string(device_information.name);
    deviceInfo["gpuClock"] = device_information.clockRate;
    deviceInfo["gpuVRAM"] = device_information.totalGlobalMem / (1024 * 1024);
    return deviceInfo;
}

nlohmann::json toJSON(ShapeDescriptor::cpu::float3 in) {
    nlohmann::json out;
    out.push_back(in.x);
    out.push_back(in.y);
    out.push_back(in.z);
    return out;
}

void ShapeBench::writeExperimentResults(const ShapeBench::ExperimentResult &results, std::filesystem::path outputBaseDirectory, bool isFinalResult, bool isPRCEnabled, const ShapeBench::ReplicationSettings& replicationSettings) {

    // 1: Initial version
    // 1.1: Added information about each filter


    nlohmann::json jsonOutput;
    jsonOutput["version"] = "1.1";

    /*jsonOutput["buildinfo"] = {};
    jsonOutput["buildinfo"]["commit"] = GitMetadata::CommitSHA1();
    jsonOutput["buildinfo"]["commitAuthor"] = GitMetadata::AuthorName();
    jsonOutput["buildinfo"]["commitDate"] = GitMetadata::CommitDate();
    jsonOutput["buildinfo"]["uncommittedChanges"] = GitMetadata::AnyUncommittedChanges();
*/
    //jsonOutput["gpuInfo"] = getGPUInfo();

    jsonOutput["experiment"]["index"] = results.experimentIndex;
    jsonOutput["experiment"]["randomSeed"] = results.experimentRandomSeed;

    jsonOutput["method"]["name"] = results.methodName;
    jsonOutput["method"]["metadata"] = results.methodMetadata;

    jsonOutput["replicatedResults"] = replicationSettings.enabled;

    jsonOutput["configuration"] = results.usedConfiguration;
    jsonOutput["computedConfiguration"] = results.usedComputedConfiguration.toJSON();

    jsonOutput["results"];
    std::cout << "    Entries in result set: " << results.vertexResults.size() << std::endl;

    for(uint32_t i = 0; i < results.vertexResults.size(); i++) {
        nlohmann::json entryJson;
        const ShapeBench::ExperimentResultsEntry& entry = results.vertexResults.at(i);

        if(entry.included) {
            entryJson["resultID"] = i;
            entryJson["modelObject"]["fractionAddedNoise"] = entry.modelObject.fractionAddedNoise;
            entryJson["modelObject"]["fractionSurfacePartiality"] = entry.modelObject.fractionSurfacePartiality;
            entryJson["modelObject"]["originalVertex"] = toJSON(entry.modelObject.originalVertexLocation.vertex);
            entryJson["modelObject"]["originalNormal"] = toJSON(entry.modelObject.originalVertexLocation.normal);
            entryJson["modelObject"]["filteredVertex"] = toJSON(entry.modelObject.filteredVertexLocation.vertex);
            entryJson["modelObject"]["filteredNormal"] = toJSON(entry.modelObject.filteredVertexLocation.normal);
            entryJson["modelObject"]["filterOutput"] = entry.modelObject.filterOutput;

            entryJson["sceneObject"]["fractionAddedNoise"] = entry.sceneObject.fractionAddedNoise;
            entryJson["sceneObject"]["fractionSurfacePartiality"] = entry.sceneObject.fractionSurfacePartiality;
            entryJson["sceneObject"]["originalVertex"] = toJSON(entry.sceneObject.originalVertexLocation.vertex);
            entryJson["sceneObject"]["originalNormal"] = toJSON(entry.sceneObject.originalVertexLocation.normal);
            entryJson["sceneObject"]["filteredVertex"] = toJSON(entry.sceneObject.filteredVertexLocation.vertex);
            entryJson["sceneObject"]["filteredNormal"] = toJSON(entry.sceneObject.filteredVertexLocation.normal);
            entryJson["sceneObject"]["filterOutput"] = entry.sceneObject.filterOutput;

            entryJson["filteredDescriptorRank"] = entry.filteredDescriptorRank;
            entryJson["meshID"] = entry.sourceVertex.meshID;
            entryJson["vertexIndex"] = entry.sourceVertex.vertexIndex;
            if (isPRCEnabled) {
                entryJson["PRC"]["distanceToNearestNeighbour"] = entry.prcMetadata.distanceToNearestNeighbour;
                entryJson["PRC"]["distanceToSecondNearestNeighbour"] = entry.prcMetadata.distanceToSecondNearestNeighbour;
                entryJson["PRC"]["modelPointMeshID"] = entry.prcMetadata.modelPointMeshID;
                entryJson["PRC"]["scenePointMeshID"] = entry.prcMetadata.scenePointMeshID;
                entryJson["PRC"]["nearestNeighbourVertexModel"] = toJSON(entry.prcMetadata.nearestNeighbourVertexModel);
                entryJson["PRC"]["nearestNeighbourVertexScene"] = toJSON(entry.prcMetadata.nearestNeighbourVertexScene);
            }

            jsonOutput["results"].push_back(entryJson);
        }


    }

    std::string experimentName = results.usedConfiguration.at("experimentsToRun").at(results.experimentIndex).at("name");
    std::string uniqueString = ShapeDescriptor::generateUniqueFilenameString();
    std::string fileName = experimentName + "-" + results.methodName + "-" + uniqueString + ".json";
    std::filesystem::path outputDirectory = outputBaseDirectory / experimentName;
    if(!isFinalResult) {
        outputDirectory /= "intermediate";
    }
    std::filesystem::create_directories(outputDirectory);
    std::filesystem::path outputFilePath = outputDirectory / fileName;

    std::ofstream outputStream(outputFilePath);
    outputStream << jsonOutput.dump(4) << std::flush;
}
