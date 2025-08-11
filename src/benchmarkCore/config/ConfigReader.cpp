#include <fstream>
#include "ConfigReader.h"


nlohmann::json ShapeBench::readConfiguration(std::filesystem::path filePath) {
    std::ifstream inputStream(filePath);
    nlohmann::json configuration = nlohmann::json::parse(inputStream);
    nlohmann::json originalConfiguration = configuration;
    if(configuration.contains("includes")) {
        std::filesystem::path containingDirectory = filePath.parent_path();
        for(std::string dependencyPathString : configuration.at("includes")) {
            nlohmann::json subConfiguration = readConfiguration(containingDirectory / dependencyPathString);
            configuration.merge_patch(subConfiguration);
        }
        // Delete includes key to ensure it does not get parsed twice
        configuration.erase("includes");
        originalConfiguration.erase("includes");
    }
    // We want the base file to override any values provided by any file it includes
    configuration.merge_patch(originalConfiguration);
    return configuration;
}