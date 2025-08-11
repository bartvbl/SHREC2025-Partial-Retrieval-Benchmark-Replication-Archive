

#include "ReplicationConfigPatcher.h"


inline void patchKey(nlohmann::json& replicationConfig, nlohmann::json& regularConfig, std::string key) {
    if(regularConfig.contains(key)) {
        replicationConfig[key].merge_patch(regularConfig.at(key));
    }
}

void ShapeBench::patchReplicationConfiguration(nlohmann::json &replicationConfig, nlohmann::json &regularConfig, const ShapeBench::ReplicationSettings& replicationSettings) {
    patchKey(replicationConfig, regularConfig, "replicationOverrides");
    patchKey(replicationConfig, regularConfig, "datasetSettings");
    patchKey(replicationConfig, regularConfig, "cacheDirectory");
    patchKey(replicationConfig, regularConfig, "resultsDirectory");
    patchKey(replicationConfig, regularConfig, "computedConfigFile");
    patchKey(replicationConfig, regularConfig, "verboseOutput");
    patchKey(replicationConfig, regularConfig, "debug_fixPartialFile");

    // Each results file only contains one active method and experiment
    // So we ensure here that in both cases only one is marked as active
    // We first enable the one active method
    for(const auto& methodEntry : replicationConfig.at("methodSettings").items()) {
        replicationConfig["methodSettings"][methodEntry.key()]["enabled"] = methodEntry.key() == replicationSettings.methodName;
    }

    // And ensure here that only the experiment being replicated is active
    for(int i = 0; i < replicationConfig.at("experimentsToRun").size(); i++) {
        replicationConfig.at("experimentsToRun").at(i).at("enabled") = i == replicationSettings.experimentIndex;
    }

    // More manual patching
    if(regularConfig.at("commonExperimentSettings").contains("intermediateSaveFrequency")) {
        replicationConfig.at("commonExperimentSettings")["intermediateSaveFrequency"] = regularConfig.at("commonExperimentSettings").at("intermediateSaveFrequency");
    }
}