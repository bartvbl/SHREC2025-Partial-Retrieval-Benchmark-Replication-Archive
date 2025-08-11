#include "FilteredMeshViewer.h"
#include "DebugRenderer.h"

void ShapeBench::viewFilteredMeshPair(const ShapeBench::FilteredMeshPair &meshes, ShapeBench::DebugRenderer& renderer) {

    renderer.drawMesh(meshes.filteredSampleMesh, {0.3, 0.67, 0.3});
    renderer.drawMesh(meshes.filteredAdditiveNoise, {0.7, 0.3, 0.3});

    for(uint32_t i = 0; i < meshes.mappedReferenceVertices.size(); i++) {
        ShapeDescriptor::cpu::float3 colour = {0.3, 0.7, 0.7};
        if(!meshes.mappedVertexIncluded.at(i)) {
            colour = {1, 0, 0};
        }
        float size = 0.1;
        ShapeDescriptor::cpu::float3 vertex = meshes.mappedReferenceVertices.at(i).vertex;
        renderer.drawLine(glm::vec3{vertex.x - size, vertex.y, vertex.z}, glm::vec3{vertex.x + size, vertex.y, vertex.z}, colour);
        renderer.drawLine(glm::vec3{vertex.x, vertex.y - size, vertex.z}, glm::vec3{vertex.x, vertex.y + size, vertex.z}, colour);
        renderer.drawLine(glm::vec3{vertex.x, vertex.y, vertex.z - size}, glm::vec3{vertex.x, vertex.y, vertex.z + size}, colour);
    }

    renderer.show("Filtered meshes");



}
