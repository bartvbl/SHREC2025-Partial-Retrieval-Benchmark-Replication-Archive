#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "MICCIDescriptor.h"

namespace ShapeBench {
    namespace micci {
        ShapeDescriptor::cpu::array<ShapeBench::PointCloudMICCIDescriptor> generateGrayscaleMICCIDescriptorsMultiRadius(
                const ShapeDescriptor::cpu::PointCloud& pointCloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const std::vector<float>& supportRadius);
    }
}