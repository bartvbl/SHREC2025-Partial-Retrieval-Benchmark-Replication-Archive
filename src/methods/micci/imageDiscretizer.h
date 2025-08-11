#pragma once

#include "MICCIDescriptor.h"

namespace ShapeBench {
    namespace micci {
        ShapeBench::MICCIDescriptor discretiseMICCIImage(
                const ShapeBench::TriangleMeshMICCIDescriptor& sourceDescriptor,
                uint32_t pointDensity);

        ShapeBench::TriangleMeshMICCIDescriptor discretiseMICCIImage(
                const ShapeBench::PointCloudMICCIDescriptor& descriptor,
                float pointDensity);

        ShapeDescriptor::gpu::array<ShapeBench::TriangleMeshMICCIDescriptor> discretiseMICCIImages(
                const ShapeDescriptor::gpu::array<ShapeBench::PointCloudMICCIDescriptor> descriptors,
                const ShapeDescriptor::gpu::array<float> supportRadii,
                float densityPerUnitArea);
    }
}