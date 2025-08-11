#pragma once

#include "nlohmann/json.hpp"
#include "benchmarkCore/BenchmarkConfiguration.h"

namespace ShapeBench {
    void patchReplicationConfiguration(nlohmann::json &replicationConfig, nlohmann::json &regularConfig, const ShapeBench::ReplicationSettings& replicationSettings);
}