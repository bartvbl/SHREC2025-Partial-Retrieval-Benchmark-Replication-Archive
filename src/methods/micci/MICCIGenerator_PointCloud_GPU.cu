#include "MICCIGenerator_PointCloud_GPU.h"
#include "tools/micciThresholdSelector/Gaussian2D.h"

#include <chrono>

__device__ __inline__ float3 transformCoordinate(const float3 &vertex, const float3 &spinImageVertex, const float3 &spinImageNormal)
{
    const float2 sineCosineAlpha = normalize(make_float2(spinImageNormal.x, spinImageNormal.y));

    const bool is_n_a_not_zero = !((abs(spinImageNormal.x) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.y) < MAX_EQUIVALENCE_ROUNDING_ERROR));

    const float alignmentProjection_n_ax = is_n_a_not_zero ? sineCosineAlpha.x : 1;
    const float alignmentProjection_n_ay = is_n_a_not_zero ? sineCosineAlpha.y : 0;

    float3 transformedCoordinate = vertex - spinImageVertex;

    const float initialTransformedX = transformedCoordinate.x;
    transformedCoordinate.x = alignmentProjection_n_ax * transformedCoordinate.x + alignmentProjection_n_ay * transformedCoordinate.y;
    transformedCoordinate.y = -alignmentProjection_n_ay * initialTransformedX + alignmentProjection_n_ax * transformedCoordinate.y;

    const float transformedNormalX = alignmentProjection_n_ax * spinImageNormal.x + alignmentProjection_n_ay * spinImageNormal.y;

    const float2 sineCosineBeta = normalize(make_float2(transformedNormalX, spinImageNormal.z));

    const bool is_n_b_not_zero = !((abs(transformedNormalX) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.z) < MAX_EQUIVALENCE_ROUNDING_ERROR));

    const float alignmentProjection_n_bx = is_n_b_not_zero ? sineCosineBeta.x : 1;
    const float alignmentProjection_n_bz = is_n_b_not_zero ? sineCosineBeta.y : 0; // discrepancy between axis here is because we are using a 2D vector on 3D axis.

    // Order matters here
    const float initialTransformedX_2 = transformedCoordinate.x;
    transformedCoordinate.x = alignmentProjection_n_bz * transformedCoordinate.x - alignmentProjection_n_bx * transformedCoordinate.z;
    transformedCoordinate.z = alignmentProjection_n_bx * initialTransformedX_2 + alignmentProjection_n_bz * transformedCoordinate.z;

    return transformedCoordinate;
}

__device__ __inline__ float2 alignWithPositiveX(const float2 &midLineDirection, const float2 &vertex)
{
    float2 transformed;
    transformed.x = midLineDirection.x * vertex.x + midLineDirection.y * vertex.y;
    transformed.y = -midLineDirection.y * vertex.x + midLineDirection.x * vertex.y;
    return transformed;
}

__global__ void computeMICCIPointCloudDescriptors(
        const ShapeDescriptor::OrientedPoint* origins,
        const float* supportRadii,
        const ShapeDescriptor::gpu::PointCloud pointCloud,
        ShapeBench::PointCloudMICCIDescriptor* outputDescriptors) {

    const ShapeDescriptor::OrientedPoint spinOrigin = origins[blockIdx.x];

    const float3 vertex = spinOrigin.vertex;
    const float3 normal = spinOrigin.normal;

    __shared__ ShapeBench::PointCloudMICCIDescriptor localDescriptor;
    for(uint32_t i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        localDescriptor.contents[i] = 0;
    }

    __syncthreads();

    const float supportRadius = supportRadii[blockIdx.x];
    const float oneOverSpinImagePixelWidth = float(spinImageWidthPixels) / supportRadius;

    for (uint32_t sampleIndex = threadIdx.x; sampleIndex < pointCloud.pointCount; sampleIndex += blockDim.x) {
        float3 samplePoint = pointCloud.vertices.at(sampleIndex);
        float3 sampleNormal = pointCloud.normals.at(sampleIndex);
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

        float2 sampleAlphaBeta = {length(make_float2(projectedVertex.x, projectedVertex.y)), projectedVertex.z};

        float floatSpinImageCoordinateX = (sampleAlphaBeta.x * oneOverSpinImagePixelWidth);
        float floatSpinImageCoordinateY = (sampleAlphaBeta.y * oneOverSpinImagePixelWidth);

        float flooredSpinImageCoordinateX = floor(floatSpinImageCoordinateX);
        float flooredSpinImageCoordinateY = floor(floatSpinImageCoordinateY);

        float locationInPixelX = floatSpinImageCoordinateX - flooredSpinImageCoordinateX;
        float locationInPixelY = floatSpinImageCoordinateY - flooredSpinImageCoordinateY;

        const float2 mean = {0.5, 0.5};
        float contributionScaleFactor = Guassian2D(mean, 0.1, {locationInPixelX, locationInPixelY});
        pointContribution *= contributionScaleFactor;

        int baseSpinImageCoordinateX = (int) flooredSpinImageCoordinateX;
        int baseSpinImageCoordinateY = (int) flooredSpinImageCoordinateY;

        const int halfSpinImageSizePixels = spinImageWidthPixels / 2;

        if (baseSpinImageCoordinateX >= 0 &&
            baseSpinImageCoordinateX < spinImageWidthPixels &&
            baseSpinImageCoordinateY >= -halfSpinImageSizePixels &&
            baseSpinImageCoordinateY < halfSpinImageSizePixels) {
            size_t valueIndex = (baseSpinImageCoordinateY + spinImageWidthPixels / 2) * spinImageWidthPixels + baseSpinImageCoordinateX;

            atomicAdd(&localDescriptor.contents[valueIndex], pointContribution);
        }
    }

    __syncthreads();

    // Copy final image into memory
    for(size_t i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        outputDescriptors[blockIdx.x].contents[i] = localDescriptor.contents[i];
    }
}



ShapeDescriptor::gpu::array<ShapeBench::PointCloudMICCIDescriptor>
ShapeBench::micci::generateGrayscaleMICCIDescriptorsMultiRadiusGPU(
        const ShapeDescriptor::gpu::PointCloud& pointCloud,
        const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
        const ShapeDescriptor::gpu::array<float> device_supportRadii) {

    // CUDA throws an error if the number of blocks to launch is 0
    // But the benchmark still expects a valid array
    // Returning here is the compromise.
    if(descriptorOrigins.length == 0) {
        return {0, nullptr};
    }

    ShapeDescriptor::gpu::array<ShapeBench::PointCloudMICCIDescriptor> device_descriptors(descriptorOrigins.length);
    size_t outputBufferSize = sizeof(ShapeBench::PointCloudMICCIDescriptor) * descriptorOrigins.length;
    checkCudaErrors(cudaMemset(device_descriptors.content, 0, outputBufferSize));

    computeMICCIPointCloudDescriptors <<<descriptorOrigins.length, 512>>>(
            descriptorOrigins.content,
            device_supportRadii.content,
            pointCloud,
            device_descriptors.content);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    return device_descriptors;
}
