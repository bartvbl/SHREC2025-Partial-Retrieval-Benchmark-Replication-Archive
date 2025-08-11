#pragma once

#include <cstdint>
#include "shapeDescriptor/shapeDescriptor.h"

namespace ShapeBench {
    ShapeDescriptor::cpu::Mesh generateUnitCubeMesh();
    ShapeDescriptor::cpu::Mesh generateUnitSphereMesh(uint32_t slices, uint32_t layers);
}