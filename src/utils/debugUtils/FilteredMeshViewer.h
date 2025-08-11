#pragma once

#include "filters/FilteredMeshPair.h"
#include "DebugRenderer.h"

namespace ShapeBench {
    void viewFilteredMeshPair(const ShapeBench::FilteredMeshPair& meshes, ShapeBench::DebugRenderer& renderer);
}