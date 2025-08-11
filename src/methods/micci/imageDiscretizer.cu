#include <bitset>
#include "imageDiscretizer.h"
#include "MICCIDescriptor.h"

template<typename InputDescriptor, typename pixelType>
ShapeBench::MICCIDescriptor convertToMICCI(const InputDescriptor& sourceDescriptor, pixelType pointDensity) {
    ShapeBench::MICCIDescriptor outDescriptor {};
    for(uint32_t row = 0; row < spinImageWidthPixels; row++) {
        std::bitset<32> batch(0);
        // pixel 0 is always 0, so we can skip it outright
        for (uint32_t pixel = 1; pixel < spinImageWidthPixels; pixel++) {
            pixelType currentNeedlePixelValue = sourceDescriptor.contents[row * spinImageWidthPixels + pixel];
            pixelType previousNeedlePixelValue = sourceDescriptor.contents[row * spinImageWidthPixels + pixel - 1];

            uint32_t estimatedIntersectionCountLeft = previousNeedlePixelValue / pointDensity;
            uint32_t estimatedIntersectionCountRight = currentNeedlePixelValue / pointDensity;

            // We may be subtracting unsigned integers here
            uint32_t imageDelta = estimatedIntersectionCountLeft >= estimatedIntersectionCountRight
                                   ? estimatedIntersectionCountLeft - estimatedIntersectionCountRight
                                   : estimatedIntersectionCountRight - estimatedIntersectionCountLeft;

            bool didIntersectionCountsChange = imageDelta != 0;

            batch[31 - (pixel % 32)] = didIntersectionCountsChange;
            if(pixel % 32 == 31) {
                size_t chunkIndex = (row * (spinImageWidthPixels / 32)) + (pixel / 32);
                outDescriptor.contents[chunkIndex] = batch.to_ulong();
            }
        }
    }
    return outDescriptor;
}

__global__ void convertToMICCI_GPU(const ShapeBench::PointCloudMICCIDescriptor* sourceDescriptors,
                               const float* supportRadii,
                               float pointDensityPerUnitArea,
                               ShapeBench::TriangleMeshMICCIDescriptor* outputDescriptors) {
    float supportRadius = supportRadii[blockIdx.x];
    float planeArea = supportRadius * supportRadius;
    float radiusAdjustedThreshold = planeArea * pointDensityPerUnitArea;

    for(uint32_t i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        outputDescriptors[blockIdx.x].contents[i] = uint32_t(sourceDescriptors[blockIdx.x].contents[i] / radiusAdjustedThreshold);
    }
}

ShapeBench::MICCIDescriptor
ShapeBench::micci::discretiseMICCIImage(const ShapeBench::TriangleMeshMICCIDescriptor &descriptor, uint32_t pointDensity) {
    return convertToMICCI<ShapeBench::TriangleMeshMICCIDescriptor, uint32_t>(descriptor, pointDensity);
}

ShapeBench::TriangleMeshMICCIDescriptor
ShapeBench::micci::discretiseMICCIImage(const ShapeBench::PointCloudMICCIDescriptor &descriptor, float pointDensity) {
    //return convertToMICCI<ShapeBench::PointCloudMICCIDescriptor, float>(descriptor, pointDensity);

    ShapeBench::TriangleMeshMICCIDescriptor outDescriptor;

    for(uint32_t i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
        outDescriptor.contents[i] = uint32_t(descriptor.contents[i] / pointDensity);
    }
    return outDescriptor;
}



ShapeDescriptor::gpu::array<ShapeBench::TriangleMeshMICCIDescriptor> ShapeBench::micci::discretiseMICCIImages(
        const ShapeDescriptor::gpu::array<ShapeBench::PointCloudMICCIDescriptor> descriptors,
        const ShapeDescriptor::gpu::array<float> supportRadii,
        float densityPerUnitArea) {
    ShapeDescriptor::gpu::array<ShapeBench::TriangleMeshMICCIDescriptor> outDescriptors(descriptors.length);

    convertToMICCI_GPU<<<descriptors.length, 128>>>(descriptors.content, supportRadii.content, densityPerUnitArea, outDescriptors.content);
    checkCudaErrors(cudaDeviceSynchronize());

    return outDescriptors;
}
