#pragma once
#include <cstdint>
#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    struct MICCIDescriptor {
        uint32_t contents[spinImageWidthPixels * spinImageWidthPixels];
    };

    struct PointCloudMICCIDescriptor {
        float contents[spinImageWidthPixels * spinImageWidthPixels];
    };

    struct TriangleMeshMICCIDescriptor {
        uint32_t contents[spinImageWidthPixels * spinImageWidthPixels];
    };
}