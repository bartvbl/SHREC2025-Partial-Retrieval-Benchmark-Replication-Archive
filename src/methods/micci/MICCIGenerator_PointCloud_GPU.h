#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "MICCIDescriptor.h"

namespace ShapeBench {
    namespace micci {
        ShapeDescriptor::gpu::array<ShapeBench::PointCloudMICCIDescriptor> generateGrayscaleMICCIDescriptorsMultiRadiusGPU(
                const ShapeDescriptor::gpu::PointCloud& pointCloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const ShapeDescriptor::gpu::array<float> supportRadii);
    }
}