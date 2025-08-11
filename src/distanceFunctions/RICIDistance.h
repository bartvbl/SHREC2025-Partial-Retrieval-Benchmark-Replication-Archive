#pragma once

#include <cstdint>

namespace ShapeBench {
    template<typename DescriptorType>
    static inline float computeRICIDistance(
            const DescriptorType& descriptor,
            const DescriptorType& otherDescriptor,
            float earlyExitThreshold) {
        uint64_t score = 0;
        uint64_t pixelSum = 0;
        uint64_t globalSquaredSum = 0;
        uint64_t squaredSum = 0;

        // Scores are computed one row at a time.
        // We differentiate between rows to ensure the final pixel of the previous row does not
        // affect the first pixel of the next one.
        for(int row = 0; row < spinImageWidthPixels; row++) {
            // Each thread processes one pixel, a warp processes therefore 32 pixels per iteration
            for (int pixel = 0; pixel < spinImageWidthPixels; pixel++) {
                uint32_t currentNeedlePixelValue = descriptor.contents[row * spinImageWidthPixels + pixel];
                uint32_t currentHaystackPixelValue = otherDescriptor.contents[row * spinImageWidthPixels + pixel];

                uint32_t previousNeedlePixelValue =   pixel > 0 ? descriptor.contents[row * spinImageWidthPixels + pixel - 1] : 0;
                uint32_t previousHaystackPixelValue = pixel > 0 ? otherDescriptor.contents[row * spinImageWidthPixels + pixel - 1] : 0;

                // The distance measure this function computes is based on deltas between pairs of pixels
                int needleDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);
                int haystackDelta = int(currentHaystackPixelValue) - int(previousHaystackPixelValue);

                // This if statement makes a massive difference in the clutter resistant performance of this method
                // It only counts least squares differences if the needle image has a change in intersection count
                // Which is usually something very specific to that object.
                if (needleDelta != 0) {
                    score += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
                    if(float(score) > earlyExitThreshold) {
                        // Basically an early exit, but need to update the last values too
                        pixel = spinImageWidthPixels;
                        row = spinImageWidthPixels;
                    }
                }
                pixelSum += currentNeedlePixelValue;
                squaredSum += haystackDelta * haystackDelta;
                globalSquaredSum += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
            }
        }

        if(pixelSum > 0) {
            return float(score);
        } else if(squaredSum > 0) {
            return float(squaredSum);
        } else {
            return float(globalSquaredSum);
        }
    }

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    template<typename DescriptorType>
    static __device__ __inline__ uint64_t computeRICIDistanceGPU(
            const DescriptorType& descriptor,
            const DescriptorType& otherDescriptor,
            float earlyExitThreshold) {
        uint64_t threadScore = 0;
        uint64_t threadPixelSum = 0;
        uint64_t threadSquaredSum = 0;
        uint64_t threadTotalSquaredSum = 0;
        const int laneIndex = threadIdx.x % 32;

        static_assert(spinImageWidthPixels % 32 == 0,
                      "This kernel assumes an image whose width is a multiple of the warp size");

        // Scores are computed one row at a time.
        // We differentiate between rows to ensure the final pixel of the previous row does not
        // affect the first pixel of the next one.
        for (int row = 0; row < spinImageWidthPixels; row++) {
            radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;
            radialIntersectionCountImagePixelType previousWarpLastHaystackPixelValue = 0;
            // Each thread processes one pixel, a warp processes therefore 32 pixels per iteration
            for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
                radialIntersectionCountImagePixelType currentNeedlePixelValue =
                        descriptor.contents[row * spinImageWidthPixels + pixel];
                radialIntersectionCountImagePixelType currentHaystackPixelValue =
                        otherDescriptor.contents[row * spinImageWidthPixels + pixel];

                // To save on memory bandwidth, we use shuffle instructions to pass around other values needed by the
                // distance computation. We first need to use some logic to determine which thread should read from which
                // other thread.
                int targetThread;
                if (laneIndex > 0) {
                    // Each thread reads from the previous one
                    targetThread = laneIndex - 1;
                }
                    // For these last two: lane index is 0. The first pixel of each row receives special treatment, as
                    // there is no pixel left of it you can compute a difference over
                else if (pixel > 0) {
                    // If pixel is not the first pixel in the row, we read the difference value from the previous iteration
                    targetThread = 31;
                } else {
                    // If the pixel is the leftmost pixel in the row, we give targetThread a dummy value that will always
                    // result in a pixel delta of zero: our own thread with ID 0.
                    targetThread = 0;
                }

                radialIntersectionCountImagePixelType threadNeedleValue = 0;
                radialIntersectionCountImagePixelType threadHaystackValue = 0;

                // Here we determine the outgoing value of the shuffle.
                // If we're the last thread in the warp, thread 0 will read from us if we're not processing the first batch
                // of 32 pixels in the row. Since in that case thread 0 will read from itself, we can simplify that check
                // into whether we are lane 31.
                if (laneIndex == 31) {
                    threadNeedleValue = previousWarpLastNeedlePixelValue;
                    threadHaystackValue = previousWarpLastHaystackPixelValue;
                } else {
                    threadNeedleValue = currentNeedlePixelValue;
                    threadHaystackValue = currentHaystackPixelValue;
                }

                // Exchange "previous pixel" values through shuffle instructions
                radialIntersectionCountImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF,
                                                                                             threadNeedleValue,
                                                                                             targetThread);
                radialIntersectionCountImagePixelType previousHaystackPixelValue = __shfl_sync(0xFFFFFFFF,
                                                                                               threadHaystackValue,
                                                                                               targetThread);

                // The distance measure this function computes is based on deltas between pairs of pixels
                int needleDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);
                int haystackDelta = int(currentHaystackPixelValue) - int(previousHaystackPixelValue);

                // This if statement makes a massive difference in the clutter resistant performance of this method
                // It only counts least squares differences if the needle image has a change in intersection count
                // Which is usually something very specific to that object.
                uint32_t pixelSquaredSum = (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
                if (needleDelta != 0) {
                    threadScore += pixelSquaredSum;
                }
                threadPixelSum += currentNeedlePixelValue * currentNeedlePixelValue;
                threadSquaredSum += haystackDelta * haystackDelta;
                threadTotalSquaredSum += pixelSquaredSum;

                // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
                previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
                previousWarpLastHaystackPixelValue = currentHaystackPixelValue;
            }
        }

        uint64_t imageScore = ShapeDescriptor::warpAllReduceSum(threadScore);
        uint64_t pixelSum = ShapeDescriptor::warpAllReduceSum(threadPixelSum);
        uint64_t squaredSum = ShapeDescriptor::warpAllReduceSum(threadSquaredSum);
        uint64_t totalSquaredSum = ShapeDescriptor::warpAllReduceSum(threadTotalSquaredSum);

        if (pixelSum > 0) {
            return imageScore;
        } else if (squaredSum > 0) {
            return squaredSum;
        } else {
            return totalSquaredSum;
        }
    }
}
#endif