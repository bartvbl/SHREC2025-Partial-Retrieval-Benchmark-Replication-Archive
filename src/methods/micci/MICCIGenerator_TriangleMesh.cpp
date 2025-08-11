#include "MICCIGenerator_TriangleMesh.h"
#include "imageDiscretizer.h"
#include "MICCIDescriptor.h"

#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>

// Borrowed from libShapeDescriptor
ShapeDescriptor::cpu::float3 transformCoordinate(const ShapeDescriptor::cpu::float3 &vertex, const ShapeDescriptor::cpu::float3 &spinImageVertex, const ShapeDescriptor::cpu::float3 &spinImageNormal);
ShapeDescriptor::cpu::float2 alignWithPositiveX(const ShapeDescriptor::cpu::float2 &midLineDirection, const ShapeDescriptor::cpu::float2 &vertex);

void ShapeBench::micci::rasteriseMICCITriangle(
        ShapeBench::TriangleMeshMICCIDescriptor& descriptor,
        std::array<ShapeDescriptor::cpu::float3, 3> vertices,
        const ShapeDescriptor::cpu::float3 &spinImageVertex,
        const ShapeDescriptor::cpu::float3 &spinImageNormal) {
    vertices[0] = transformCoordinate(vertices[0], spinImageVertex, spinImageNormal);
    vertices[1] = transformCoordinate(vertices[1], spinImageVertex, spinImageNormal);
    vertices[2] = transformCoordinate(vertices[2], spinImageVertex, spinImageNormal);

    // Sort vertices by z-coordinate

    int minIndex = 0;
    int midIndex = 1;
    int maxIndex = 2;
    int _temp;

    if (vertices[minIndex].z > vertices[midIndex].z)
    {
        _temp = minIndex;
        minIndex = midIndex;
        midIndex = _temp;
    }
    if (vertices[minIndex].z > vertices[maxIndex].z)
    {
        _temp = minIndex;
        minIndex = maxIndex;
        maxIndex = _temp;
    }
    if (vertices[midIndex].z > vertices[maxIndex].z)
    {
        _temp = midIndex;
        midIndex = maxIndex;
        maxIndex = _temp;
    }

    const ShapeDescriptor::cpu::float3 minVector = vertices[minIndex];
    const ShapeDescriptor::cpu::float3 midVector = vertices[midIndex];
    const ShapeDescriptor::cpu::float3 maxVector = vertices[maxIndex];

    // Calculate deltas

    const ShapeDescriptor::cpu::float3 deltaMinMid = midVector - minVector;
    const ShapeDescriptor::cpu::float3 deltaMidMax = maxVector - midVector;
    const ShapeDescriptor::cpu::float3 deltaMinMax = maxVector - minVector;

    // Horizontal triangles are most likely not to register, and cause zero divisions, so it's easier to just get rid of them.
    if (deltaMinMax.z < MAX_EQUIVALENCE_ROUNDING_ERROR)
    {
        return;
    }

    // Step 6: Calculate centre line
    const float centreLineFactor = deltaMinMid.z / deltaMinMax.z;
    const ShapeDescriptor::cpu::float2 centreLineDelta = centreLineFactor * ShapeDescriptor::cpu::float2{deltaMinMax.x, deltaMinMax.y};
    const ShapeDescriptor::cpu::float2 centreLineDirection = centreLineDelta - ShapeDescriptor::cpu::float2{deltaMinMid.x, deltaMinMid.y};
    const ShapeDescriptor::cpu::float2 centreDirection = normalize(centreLineDirection);

    // Step 7: Rotate coordinates around origin
    // From here on out, variable names follow these conventions:
    // - X: physical relative distance to closest point on intersection line
    // - Y: Distance from origin
    const ShapeDescriptor::cpu::float2 minXY = alignWithPositiveX(centreDirection, ShapeDescriptor::cpu::float2{minVector.x, minVector.y});
    const ShapeDescriptor::cpu::float2 midXY = alignWithPositiveX(centreDirection, ShapeDescriptor::cpu::float2{midVector.x, midVector.y});
    const ShapeDescriptor::cpu::float2 maxXY = alignWithPositiveX(centreDirection, ShapeDescriptor::cpu::float2{maxVector.x, maxVector.y});

    const ShapeDescriptor::cpu::float2 deltaMinMidXY = midXY - minXY;
    const ShapeDescriptor::cpu::float2 deltaMidMaxXY = maxXY - midXY;
    const ShapeDescriptor::cpu::float2 deltaMinMaxXY = maxXY - minXY;

    // Step 8: For each row, do interpolation
    // And ensure we only rasterise within bounds

    // Adding 0.5 causes the floor function to behave like a round() function that does not round to zero
    // The difference in + and - is to cause pixels to only be drawn if they cross a half pixel (0.5 pixel) line in the image
    // Only then is there a chance that a circle intersects
    const int minPixels = std::floor(minVector.z + 0.5);
    const int maxPixels = std::floor(maxVector.z - 0.5);

    const int halfHeight = spinImageWidthPixels / 2;

    // Filter out job batches with no work in them
    if((minPixels < -halfHeight && maxPixels < -halfHeight) ||
       (minPixels >= halfHeight && maxPixels >= halfHeight)) {
        return;
    }

    const int startRowIndex = std::max<int>(-halfHeight, minPixels);
    const int endRowIndex = std::min<int>(halfHeight - 1, maxPixels);

    for(int pixelY = startRowIndex; pixelY <= endRowIndex; pixelY++)
    {
        // Verified: this should be <=, because it fails for the cube tests case
        const bool isBottomSection = (float(pixelY) + 0.5) <= midVector.z;

        // Technically I can rewrite this into two separate loops
        // However, that would increase the thread divergence
        // I believe this is the best option
        const float shortDeltaVectorZ = isBottomSection ? deltaMinMid.z : deltaMidMax.z;
        const float shortVectorStartZ = isBottomSection ? minVector.z : midVector.z;
        const ShapeDescriptor::cpu::float2 shortVectorStartXY = isBottomSection ? minXY : midXY;
        const ShapeDescriptor::cpu::float2 shortTransformedDelta = isBottomSection ? deltaMinMidXY : deltaMidMaxXY;

        // Circles are located at 0.5, 0.5 in each pixel. So to convert from pixel to world space we need to add that offset back in
        const float zLevel = float(pixelY) + 0.5;
        const float longDistanceInTriangle = zLevel - minVector.z;
        const float longInterpolationFactor = longDistanceInTriangle / deltaMinMax.z;
        const float shortDistanceInTriangle = zLevel - shortVectorStartZ;
        const float shortInterpolationFactor = (shortDeltaVectorZ == 0) ? 1.0f : shortDistanceInTriangle / shortDeltaVectorZ;
        // Set value to 1 because we want to avoid a zero division, and we define the job Z level to be at its maximum height

        const uint32_t pixelYCoordinate = pixelY + halfHeight;
        // Avoid overlap situations, only rasterise is the interpolation factors are valid
        if (longDistanceInTriangle > 0 && shortDistanceInTriangle > 0)
        {
            // y-coordinates of both interpolated values are always equal. As such we only need to interpolate that direction once.
            // They must be equal because we have aligned the direction of the horizontal-triangle plane with the x-axis.
            const float intersectionY = minXY.y + (longInterpolationFactor * deltaMinMaxXY.y);
            // The other two x-coordinates are interpolated separately.
            const float intersection1X = shortVectorStartXY.x + (shortInterpolationFactor * shortTransformedDelta.x);
            const float intersection2X = minXY.x + (longInterpolationFactor * deltaMinMaxXY.x);

            const float intersection1Distance = length(ShapeDescriptor::cpu::float2{intersection1X, intersectionY});
            const float intersection2Distance = length(ShapeDescriptor::cpu::float2{intersection2X, intersectionY});

            // Check < 0 because we omit the case where there is exactly one point with a double intersection
            const bool hasDoubleIntersection = (intersection1X * intersection2X) < 0;

            // If both values are positive or both values are negative, there is no double intersection.
            // iF the signs of the two values is different, the result will be negative or 0.
            // Having different signs implies the existence of double intersections.
            const float doubleIntersectionDistance = std::abs(intersectionY);

            const float minDistance = intersection1Distance < intersection2Distance ? intersection1Distance : intersection2Distance;
            const float maxDistance = intersection1Distance > intersection2Distance ? intersection1Distance : intersection2Distance;

            // Round places the cutoff point for the pixel at 0.5, 0.5. RICI itself has it "somewhere in the pixel"
            uint32_t rowStartPixels = std::floor(minDistance + 0.5);
            uint32_t rowEndPixels = std::floor(maxDistance + 0.5);

            // Ensure we are only rendering within bounds
            rowStartPixels = std::min<uint32_t>(spinImageWidthPixels, rowStartPixels);
            rowEndPixels = std::min<uint32_t>(spinImageWidthPixels, rowEndPixels);

            // Step 9: Fill pixels
            if (hasDoubleIntersection)
            {
                // since this is an absolute value, it can only be 0 or higher.
                const uint32_t jobDoubleIntersectionStartPixels = std::floor(doubleIntersectionDistance + 0.5);

                // rowStartPixels must already be in bounds, and doubleIntersectionStartPixels can not be smaller than 0.
                // Hence the values in this loop are in-bounds.
                for (uint32_t jobX = jobDoubleIntersectionStartPixels; jobX < rowStartPixels; jobX++)
                {
                    assert(jobX < spinImageWidthPixels);
                    // Increment pixel by 2 because 2 intersections occurred.
                    descriptor.contents[pixelYCoordinate * spinImageWidthPixels + jobX] += 2;
                }
            }

            // It's imperative the condition of this loop is a < comparison
            for (uint32_t jobX = rowStartPixels; jobX < rowEndPixels; jobX++)
            {
                assert(jobX < spinImageWidthPixels);
                descriptor.contents[pixelYCoordinate * spinImageWidthPixels + jobX] += 1;
            }
        }
    }
}


ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor>
ShapeBench::micci::generateGrayscaleMICCIDescriptorMultiRadius(const ShapeDescriptor::cpu::Mesh &mesh,
                                                       const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                                                       const std::vector<float> &supportRadii) {
    assert(descriptorOrigins.length == supportRadii.size());
    ShapeDescriptor::cpu::array<ShapeBench::TriangleMeshMICCIDescriptor> descriptors(descriptorOrigins.length);

    for(uint32_t descriptorIndex = 0; descriptorIndex < descriptorOrigins.length; descriptorIndex++) {
        float supportRadius = supportRadii.at(descriptorIndex);
        float scaleFactor = float(spinImageWidthPixels) / supportRadius;

        ShapeDescriptor::cpu::float3 spinImageVertex = descriptorOrigins.content[descriptorIndex].vertex * scaleFactor;
        ShapeDescriptor::cpu::float3 spinImageNormal = descriptorOrigins.content[descriptorIndex].normal;

        ShapeBench::TriangleMeshMICCIDescriptor continuousDescriptor{};
        std::fill(continuousDescriptor.contents, continuousDescriptor.contents + (sizeof(ShapeBench::TriangleMeshMICCIDescriptor) / sizeof(uint32_t)), 0);

        const size_t triangleCount = mesh.vertexCount / 3;
        for (int triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
        {
            std::array<ShapeDescriptor::cpu::float3, 3> vertices;

            vertices[0] = mesh.vertices[3 * triangleIndex + 0] * scaleFactor;
            vertices[1] = mesh.vertices[3 * triangleIndex + 1] * scaleFactor;
            vertices[2] = mesh.vertices[3 * triangleIndex + 2] * scaleFactor;

            rasteriseMICCITriangle(continuousDescriptor, vertices, spinImageVertex, spinImageNormal);
        }

        descriptors.content[descriptorIndex] = continuousDescriptor;
    }

    // ShapeBench::micci::discretiseMICCIImage(continuousDescriptor, discretisationThreshold)

    return descriptors;
}
