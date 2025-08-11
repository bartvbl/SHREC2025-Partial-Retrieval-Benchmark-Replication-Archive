#pragma once

#include "MICCIDescriptor.h"

namespace ShapeBench {
    namespace micci {
        ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> generateGrayscaleMICCIDescriptorMultiRadius(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const std::vector<float>& supportRadius);

        void rasteriseMICCITriangle(
                ShapeBench::TriangleMeshMICCIDescriptor& descriptor,
                std::array<ShapeDescriptor::cpu::float3, 3> vertices,
                const ShapeDescriptor::cpu::float3 &spinImageVertex,
                const ShapeDescriptor::cpu::float3 &spinImageNormal);
    }
}