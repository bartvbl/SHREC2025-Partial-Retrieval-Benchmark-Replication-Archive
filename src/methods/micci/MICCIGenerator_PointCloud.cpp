#include "MICCIGenerator_PointCloud.h"

#include "MICCIGenerator_TriangleMesh.h"
#include "imageDiscretizer.h"
#include "tools/micciThresholdSelector/Gaussian2D.h"

#include <cassert>
#include <chrono>

ShapeDescriptor::cpu::float3 transformCoordinate(const ShapeDescriptor::cpu::float3 &vertex, const ShapeDescriptor::cpu::float3 &spinImageVertex, const ShapeDescriptor::cpu::float3 &spinImageNormal);
ShapeDescriptor::cpu::float2 alignWithPositiveX(const ShapeDescriptor::cpu::float2 &midLineDirection, const ShapeDescriptor::cpu::float2 &vertex);



void createDescriptor(
        const ShapeDescriptor::OrientedPoint spinOrigin,
        const ShapeDescriptor::cpu::PointCloud& pointCloud,
        ShapeBench::PointCloudMICCIDescriptor& descriptor,
        float supportRadius) {

    const float3 vertex = spinOrigin.vertex;
    const float3 normal = spinOrigin.normal;

    for (int sampleIndex = 0; sampleIndex < pointCloud.pointCount; sampleIndex++) {
        float3 samplePoint = pointCloud.vertices[sampleIndex];

        if(length(samplePoint - vertex) > supportRadius) {
            continue;
        }

        float3 sampleNormal = pointCloud.normals[sampleIndex];
        sampleNormal = normalize(sampleNormal);

        float3 projectedVertex = transformCoordinate(samplePoint, vertex, normal);
        float3 projectedNormal = transformCoordinate(sampleNormal, {0, 0, 0}, normal);
        projectedNormal = normalize(projectedNormal);

        // We discard the third dimension here
        float2 pointSampleDirectionVector = normalize(make_float2(projectedVertex.x, projectedVertex.y));
        float2 circleTangent = make_float2(-pointSampleDirectionVector.y, pointSampleDirectionVector.x);
        float2 relativeSampleNormal = make_float2(projectedNormal.x, projectedNormal.y);

        float pointContribution = dot(circleTangent, relativeSampleNormal);

        pointContribution = /* 1 - */ abs(pointContribution);

        // If the point is on the image's central axis, ignore the sample contribution
        if (projectedVertex.x == 0 && projectedVertex.y == 0) {
            pointContribution = 0;
        }

        ShapeDescriptor::cpu::float2 sampleAlphaBeta = {length(make_float2(projectedVertex.x, projectedVertex.y)),
                                                        projectedVertex.z};

        const float oneOverSpinImagePixelWidth = float(spinImageWidthPixels) / supportRadius;
        float floatSpinImageCoordinateX = (sampleAlphaBeta.x * oneOverSpinImagePixelWidth);
        float floatSpinImageCoordinateY = (sampleAlphaBeta.y * oneOverSpinImagePixelWidth);

        float flooredSpinImageCoordinateX = floor(floatSpinImageCoordinateX);
        float flooredSpinImageCoordinateY = floor(floatSpinImageCoordinateY);

        float locationInPixelX = floatSpinImageCoordinateX - flooredSpinImageCoordinateX;
        float locationInPixelY = floatSpinImageCoordinateY - flooredSpinImageCoordinateY;

        const ShapeDescriptor::cpu::float2 mean = {0.5, 0.5};
        float contributionScaleFactor = Guassian2D(mean, 0.1, {locationInPixelX, locationInPixelY});
        pointContribution *= contributionScaleFactor;
        ShapeDescriptor::cpu::float2 delta = ShapeDescriptor::cpu::float2{locationInPixelX, locationInPixelY} - mean;
        float distanceToCenter = length(delta);
        //if(distanceToCenter > 0.1) { pointContribution = 0; }


        int baseSpinImageCoordinateX = (int) flooredSpinImageCoordinateX;
        int baseSpinImageCoordinateY = (int) flooredSpinImageCoordinateY;

        const int halfSpinImageSizePixels = spinImageWidthPixels / 2;

        if (baseSpinImageCoordinateX >= 0 &&
            baseSpinImageCoordinateX < spinImageWidthPixels &&
            baseSpinImageCoordinateY >= -halfSpinImageSizePixels &&
            baseSpinImageCoordinateY < halfSpinImageSizePixels) {
            size_t valueIndex = size_t(
                    (baseSpinImageCoordinateY + 0 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                    baseSpinImageCoordinateX + 0);

            descriptor.contents[valueIndex] += pointContribution;
        }
    }
}



ShapeDescriptor::cpu::array<ShapeBench::PointCloudMICCIDescriptor>
ShapeBench::micci::generateGrayscaleMICCIDescriptorsMultiRadius(const ShapeDescriptor::cpu::PointCloud &pointCloud,
                                                       const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                                                       const std::vector<float> &supportRadius) {
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t imageCount = descriptorOrigins.length;
    ShapeDescriptor::cpu::array<ShapeBench::PointCloudMICCIDescriptor> descriptors(imageCount);

    for(uint32_t descriptorIndex = 0; descriptorIndex < imageCount; descriptorIndex++) {
        ShapeBench::PointCloudMICCIDescriptor continuousDescriptor{};
        std::fill(continuousDescriptor.contents, continuousDescriptor.contents + spinImageWidthPixels * spinImageWidthPixels, 0);
        createDescriptor(descriptorOrigins.content[descriptorIndex], pointCloud, continuousDescriptor, supportRadius.at(descriptorIndex));
        descriptors.content[descriptorIndex] = continuousDescriptor;
    }

    // ShapeBench::micci::discretiseMICCIImage(continuousDescriptor, discretisationThreshold);

    return descriptors;
}
