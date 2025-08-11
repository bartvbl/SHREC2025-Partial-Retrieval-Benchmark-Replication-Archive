#pragma once

class AdditiveNoiseCache;

#include <shapeDescriptor/shapeDescriptor.h>
#include "nlohmann/json.hpp"
#include "benchmarkCore/config/ComputedConfig.h"
#include "dataset/Dataset.h"
#include "AdditiveNoiseCache.h"
#include "filters/Filter.h"

namespace ShapeBench {

    class AdditiveNoiseFilter : public ShapeBench::Filter {
        AdditiveNoiseCache additiveNoiseCache;


    public:
        void init(const nlohmann::json& config, bool invalidateCaches) override;
        void destroy() override;
        void saveCaches(const nlohmann::json& config) override;

        FilterOutput apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene, const Dataset &dataset,
                           ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) override;

        FilterOutput applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                 ShapeBench::FilteredMeshPair &scene, const Dataset &dataset,
                                 ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) override;

        constexpr bool mustBeAppliedOnBothMeshes() override {
            return false;
        }
        constexpr bool appliesNonrigidTransformation() override {
            return false;
        }
        const std::string getFilterName() const override;
    };

    std::vector<ShapeBench::Orientation> runPhysicsSimulation(ShapeBench::AdditiveNoiseFilterSettings settings, const std::vector<ShapeDescriptor::cpu::Mesh>& meshes);
}
