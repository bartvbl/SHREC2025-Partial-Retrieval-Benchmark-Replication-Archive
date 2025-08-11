#include "arrrgh.hpp"
#include "dataset/Dataset.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include "benchmarkCore/MissingBenchmarkConfigurationException.h"
#include "benchmarkCore/constants.h"
#include "methods/QUICCIMethod.h"
#include "methods/SIMethod.h"
#include "filters/additiveNoise/additiveNoiseFilter.h"
#include "benchmarkCore/experimentRunner.h"
#include "methods/3DSCMethod.h"
#include "methods/RoPSMethod.h"
#include "methods/RICIMethod.h"
#include "methods/USCMethod.h"
#include "methods/SHOTMethod.h"
#include "benchmarkCore/BenchmarkConfiguration.h"
#include "dataset/DatasetLoader.h"
#include "methods/GEDIMethod.h"
#include "methods/COPSMethod.h"
#include "methods/MICCIMethod_triangles.h"
#include "replication/ReplicationConfigPatcher.h"
#include "benchmarkCore/config/ConfigReader.h"
#include "methods/MICCIMethod_pointCloud.h"

int main(int argc, const char** argv) {
    const std::string replicationDisabledString = "UNSPECIFIED";

    arrrgh::parser parser("shapebench", "Benchmark tool for 3D local shape descriptors");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message", 'h', arrrgh::Optional, false);
    const auto& configurationFile = parser.add<std::string>(
            "configuration-file", "Location of the file from which to read the experimental configuration", '\0', arrrgh::Optional, "../cfg/config.json");
    const auto& replicateFromConfiguration = parser.add<std::string>(
            "replicate-results-file", "Path to a results file that should be replicated. Enables replication mode. Overwrites the configuration specified in the --configuration-file parameter, except for a few specific entries that are system specific.", '\0', arrrgh::Optional, replicationDisabledString);

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

    ShapeBench::BenchmarkConfiguration setup;
    setup.replicationFilePath = replicateFromConfiguration.value();
    setup.configurationFilePath = configurationFile.value();


    // --- Initialise replication mode ---
    if(setup.replicationFilePath != replicationDisabledString) {
        if(!std::filesystem::exists(setup.replicationFilePath)) {
            throw std::runtime_error("The specified results file does not appear to exist. Exiting.");
        }

        std::cout << "Replication mode enabled." << std::endl;
        setup.replicationSettings.enabled = true;
        std::cout << "    This will mostly override the specified configuration in the main configuration file." << std::endl;
        std::cout << "Reading results file.." << std::endl;
        std::ifstream inputStream(setup.replicationFilePath);
        nlohmann::json resultsFileContents = nlohmann::json::parse(inputStream);
        setup.configuration = resultsFileContents.at("configuration");
        setup.computedConfiguration = resultsFileContents.at("computedConfiguration");
        setup.replicationSettings.methodName = resultsFileContents.at("method").at("name");
        setup.replicationSettings.experimentIndex = resultsFileContents.at("experiment").at("index");
        setup.replicationSettings.experimentResults = resultsFileContents.at("results");

        if(resultsFileContents.contains("replicatedResults") && resultsFileContents.at("replicatedResults")) {
            throw std::runtime_error("The specified results file contains the results of a run that replicated another set of results. "
                                     "These results are not guaranteed to be replicable, because they for example may have only recomputed a subset of results. "
                                     "Please replicate the original results file instead.");
        }
    }



    // --- Compute experiment configuration ---
    std::cout << "Reading configuration.." << std::endl;
    const std::filesystem::path configurationFileLocation(configurationFile.value());
    bool configurationFileExists = std::filesystem::exists(configurationFileLocation);
    if(!configurationFileExists && !setup.replicationSettings.enabled) {
        throw std::runtime_error("The specified configuration file was not found at: " + std::filesystem::absolute(configurationFileLocation).string());
    }

    nlohmann::json mainConfigFileContents;
    if(configurationFileExists) {
        mainConfigFileContents = ShapeBench::readConfiguration(configurationFile.value());
    } else {
        std::cout << "    WARNING: the configuration file located at " << configurationFile.value() << " was not found." << std::endl;
        std::cout << "    Since replication mode is enabled, the configuration from the results file will be used instead." << std::endl;
        std::cout << "    This may cause the benchmark to crash if things like the dataset files are located in a different location." << std::endl;
        std::cout << "    If so, you should try specifying a configuration file to use using the --configuration-file parameter." << std::endl << std::endl;
    }

    if(!setup.replicationSettings.enabled) {
        setup.configuration = mainConfigFileContents;
        if(!setup.configuration.contains("cacheDirectory")) {
            throw ShapeBench::MissingBenchmarkConfigurationException("cacheDirectory");
        }
    } else if(configurationFileExists) {
        // For the purposes of replication, some configuration entries need to be adjusted to the
        // environment where the results are replicated. This is done by copying all relevant configuration
        // entries, and overwriting them where relevant.
        ShapeBench::patchReplicationConfiguration(setup.configuration, mainConfigFileContents, setup.replicationSettings);
    }



    // --- Compute and load dataset ---
    setup.dataset = ShapeBench::computeOrLoadCache(setup);

    double datasetSizeLimitGB = setup.configuration.at("datasetSettings").at("cacheSizeLimitGB");
    uint64_t datasetSizeLimitBytes = uint64_t(datasetSizeLimitGB * 1024.0 * 1024.0 * 1024.0);
    std::filesystem::path datasetCacheDirectory = setup.configuration.at("datasetSettings").at("compressedRootDir");
    std::string datasetDownloadBaseURL = setup.configuration.at("datasetSettings").at("objaverseDownloadBaseURL");
    bool verifyIntegrityOfDownloadedFiles = setup.configuration.at("datasetSettings").at("verifyFileIntegrity");
    ShapeBench::LocalDatasetCache* fileCache = new ShapeBench::LocalDatasetCache(datasetCacheDirectory, datasetDownloadBaseURL,  datasetSizeLimitBytes, verifyIntegrityOfDownloadedFiles);



    // --- Run experiments ---
    const nlohmann::json& methodSettings = setup.configuration.at("methodSettings");

    // ADD METHODS TO TEST HERE BY DUPLICATING THE CALL TO testMethod() ALONG WITH ITS SURROUNDING IF STATEMENT

    if(methodSettings.contains(ShapeBench::GEDIMethod::getName()) && methodSettings.at(ShapeBench::GEDIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::GEDIMethod, ShapeBench::GEDIMethod::DescriptorType>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::COPSMethod::getName()) && methodSettings.at(ShapeBench::COPSMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::COPSMethod, ShapeBench::COPSMethod::DescriptorType>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::MICCIMethod_triangles::getName()) && methodSettings.at(ShapeBench::MICCIMethod_triangles::getName()).at("enabled")) {
        testMethod<ShapeBench::MICCIMethod_triangles, ShapeBench::TriangleMeshMICCIDescriptor>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::QUICCIMethod::getName()) && methodSettings.at(ShapeBench::QUICCIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::QUICCIMethod, ShapeDescriptor::QUICCIDescriptor>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::MICCIMethod_pointCloud::getName()) && methodSettings.at(ShapeBench::MICCIMethod_pointCloud::getName()).at("enabled")) {
        testMethod<ShapeBench::MICCIMethod_pointCloud, ShapeBench::TriangleMeshMICCIDescriptor>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::RICIMethod::getName()) && methodSettings.at(ShapeBench::RICIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::RICIMethod, ShapeDescriptor::RICIDescriptor>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::RoPSMethod::getName()) && methodSettings.at(ShapeBench::RoPSMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::RoPSMethod, ShapeDescriptor::RoPSDescriptor>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::SIMethod::getName()) && methodSettings.at(ShapeBench::SIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::SIMethod, ShapeDescriptor::SpinImageDescriptor>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::USCMethod::getName()) && methodSettings.at(ShapeBench::USCMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::USCMethod, ShapeDescriptor::UniqueShapeContextDescriptor>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::ShapeContextMethod::getName()) && methodSettings.at(ShapeBench::ShapeContextMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::ShapeContextMethod, ShapeDescriptor::ShapeContextDescriptor>(setup, fileCache);
    }
    if(methodSettings.contains(ShapeBench::SHOTMethod<>::getName()) && methodSettings.at(ShapeBench::SHOTMethod<>::getName()).at("enabled")) {
        testMethod<ShapeBench::SHOTMethod<>, ShapeDescriptor::SHOTDescriptor<>>(setup, fileCache);
    }

    // Disabled and WIP. Horrendously slow.
    //testMethod<ShapeBench::FPFHMethod, ShapeDescriptor::FPFHDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);

    delete fileCache;
}