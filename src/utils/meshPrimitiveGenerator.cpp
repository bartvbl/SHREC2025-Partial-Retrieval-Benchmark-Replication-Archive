

#include <cmath>
#include "meshPrimitiveGenerator.h"

ShapeDescriptor::cpu::Mesh ShapeBench::generateUnitCubeMesh() {
    ShapeDescriptor::cpu::Mesh cubeMesh(6 * 2 * 3);

    const std::array<ShapeDescriptor::cpu::float3, 8> vertices = {
            ShapeDescriptor::cpu::float3{ 1.0f, -1.0f, -1.0f},
            ShapeDescriptor::cpu::float3{ 1.0f, -1.0f,  1.0f},
            ShapeDescriptor::cpu::float3{-1.0f, -1.0f,  1.0f},
            ShapeDescriptor::cpu::float3{-1.0f, -1.0f, -1.0f},
            ShapeDescriptor::cpu::float3{ 1.0f,  1.0f, -1.0f},
            ShapeDescriptor::cpu::float3{ 1.0f,  1.0f,  1.0f},
            ShapeDescriptor::cpu::float3{-1.0f,  1.0f,  1.0f},
            ShapeDescriptor::cpu::float3{-1.0f,  1.0f, -1.0f}
    };

    const std::array<ShapeDescriptor::cpu::float3, 6> normals = {
            ShapeDescriptor::cpu::float3{0.0f, -1.0f, 0.0f},
            ShapeDescriptor::cpu::float3{0.0f, 1.0f, 0.0f},
            ShapeDescriptor::cpu::float3{1.0f, 0.0f, 0.0f},
            ShapeDescriptor::cpu::float3{-0.0f, 0.0f, 1.0f},
            ShapeDescriptor::cpu::float3{-1.0f, -0.0f, -0.0f},
            ShapeDescriptor::cpu::float3{0.0f, 0.0f, -1.0f}
    };

    const std::array<uint32_t, 6 * 2 * 3> vertexIndices = {
            1, 2, 3,    7, 6, 5,    4, 5, 1,    5, 6, 2,    2, 6, 7,    0, 3, 7,
            0, 1, 3,    4, 7, 5,    0, 4, 1,    1, 5, 2,    3, 2, 7,    4, 0, 7
    };

    const std::array<uint32_t, 6 * 2 * 3> normalIndices = {
            0, 0, 0,    1, 1, 1,    2, 2, 2,    3, 3, 3,    4, 4, 4,    5, 5, 5,
            0, 0, 0,    1, 1, 1,    2, 2, 2,    3, 3, 3,    4, 4, 4,    5, 5, 5
    };

    for(uint32_t i = 0; i < cubeMesh.vertexCount; i++) {
        cubeMesh.vertices[i] = vertices.at(vertexIndices.at(i));
        cubeMesh.normals[i] = normals.at(normalIndices.at(i));
    }

    return cubeMesh;
}

ShapeDescriptor::cpu::Mesh ShapeBench::generateUnitSphereMesh(uint32_t slices, uint32_t layers) {

    const uint32_t triangleCount = slices * layers * 2;
    ShapeDescriptor::cpu::Mesh mesh(3 * triangleCount);

    // Slices require us to define a full revolution worth of triangles.
    // Layers only requires angle varying between the bottom and the top (a layer only covers half a circle worth of angles)
    const float anglePerLayer = float(M_PI / float(layers));
    const float anglePerSlice = float(2.0 * M_PI / float(slices));

    uint32_t nextStartingIndex = 0;

    // Constructing the sphere one layer at a time
    for (uint32_t layer = 0; layer < layers; layer++) {
        uint32_t nextLayer = layer + 1;

        // Angles between the vector pointing to any point on a particular layer and the negative z-axis
        float currentAngleZ = anglePerLayer * float(layer);
        float nextAngleZ = anglePerLayer * float(nextLayer);

        // All coordinates within a single layer share z-coordinates.
        // So we can calculate those of the current and subsequent layer here.
        float currentZ = -std::cos(currentAngleZ);
        float nextZ = -std::cos(nextAngleZ);

        // The row of vertices forms a circle around the vertical diagonal (z-axis) of the sphere.
        // These radii are also constant for an entire layer, so we can precalculate them.
        float radius = std::sin(currentAngleZ);
        float nextRadius = std::sin(nextAngleZ);

        // Now we can move on to constructing individual slices within a layer
        for (uint32_t slice = 0; slice < slices; slice++) {

            // The direction of the start and the end of the slice in the xy-plane
            float currentSliceAngleDegrees = float(slice) * anglePerSlice;
            float nextSliceAngleDegrees = float(slice + 1) * anglePerSlice;

            // Determining the direction vector for both the start and end of the slice
            float currentDirectionX = std::cos(currentSliceAngleDegrees);
            float currentDirectionY = std::sin(currentSliceAngleDegrees);

            float nextDirectionX = std::cos(nextSliceAngleDegrees);
            float nextDirectionY = std::sin(nextSliceAngleDegrees);

            mesh.vertices[nextStartingIndex + 0] = {radius * currentDirectionX,     radius * currentDirectionY,     currentZ};
            mesh.vertices[nextStartingIndex + 1] = {radius * nextDirectionX,        radius * nextDirectionY,        currentZ};
            mesh.vertices[nextStartingIndex + 2] = {nextRadius * nextDirectionX,    nextRadius * nextDirectionY,    nextZ};
            mesh.vertices[nextStartingIndex + 3] = {radius * currentDirectionX,     radius * currentDirectionY,     currentZ};
            mesh.vertices[nextStartingIndex + 4] = {nextRadius * nextDirectionX,    nextRadius * nextDirectionY,    nextZ};
            mesh.vertices[nextStartingIndex + 5] = {nextRadius * currentDirectionX, nextRadius * currentDirectionY, nextZ};

            // Unit sphere means vertices == normals
            mesh.normals[nextStartingIndex + 0] = mesh.vertices[nextStartingIndex + 0];
            mesh.normals[nextStartingIndex + 1] = mesh.vertices[nextStartingIndex + 1];
            mesh.normals[nextStartingIndex + 2] = mesh.vertices[nextStartingIndex + 2];
            mesh.normals[nextStartingIndex + 3] = mesh.vertices[nextStartingIndex + 3];
            mesh.normals[nextStartingIndex + 4] = mesh.vertices[nextStartingIndex + 4];
            mesh.normals[nextStartingIndex + 5] = mesh.vertices[nextStartingIndex + 5];

            nextStartingIndex += 6;
        }
    }

    return mesh;
}
