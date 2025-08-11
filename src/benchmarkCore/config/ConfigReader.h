#pragma once

#include "nlohmann/json.hpp"

namespace ShapeBench {
    nlohmann::json readConfiguration(std::filesystem::path filePath);
}