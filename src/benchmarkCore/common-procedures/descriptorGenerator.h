#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "dataset/Dataset.h"
#include "nlohmann/json.hpp"
#include "benchmarkCore/Batch.h"
#include "supportRadiusEstimation/SupportRadiusEstimation.h"
#include "pointCloudSampler.h"
#include "utils/debugUtils/DebugRenderer.h"

namespace ShapeBench {
    template<typename DescriptorMethod, typename DescriptorType>
    void computeDescriptors(
            const ShapeDescriptor::cpu::Mesh& mesh,
            const ShapeDescriptor::cpu::PointCloud& pointCloud,
            const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
            const std::vector<bool>& whichDescriptorsToCompute,
            const nlohmann::json &config,
            const std::vector<float>& supportRadii,
            uint64_t randomSeed,
            std::vector<DescriptorType>& outputDescriptors) {
        ShapeDescriptor::cpu::array<DescriptorType> descriptors;

        assert(whichDescriptorsToCompute.size() == descriptorOrigins.length);
        assert(supportRadii.size() == descriptorOrigins.length);
        assert(outputDescriptors.size() == descriptorOrigins.length);

        bool allDescriptorsActive = std::all_of(whichDescriptorsToCompute.begin(), whichDescriptorsToCompute.end(), [](bool b){ return b; });

        std::vector<uint32_t> originalDescriptorIndices;
        std::vector<float> condensedSupportRadii;
        std::vector<ShapeDescriptor::OrientedPoint> condensedDescriptorOrigins;

        // Condense which descriptors we want to compute, but only if we need to
        if(!allDescriptorsActive) {
            originalDescriptorIndices.reserve(descriptorOrigins.length);
            condensedSupportRadii.reserve(descriptorOrigins.length);

            for(uint32_t i = 0; i < descriptorOrigins.length; i++) {
                if(whichDescriptorsToCompute.at(i)) {
                    originalDescriptorIndices.push_back(i);
                    condensedSupportRadii.push_back(supportRadii.at(i));
                    condensedDescriptorOrigins.push_back(descriptorOrigins.content[i]);
                }
            }
            std::cout << "Condensed " << supportRadii.size() << " into " << condensedDescriptorOrigins.size() << " points." << std::endl;
        }

        const std::vector<float>& potentiallyCondensedSupportRadii = allDescriptorsActive ? supportRadii : condensedSupportRadii;
        const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> potentiallyCondensedDescriptorOrigins = allDescriptorsActive
                ? descriptorOrigins : ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>{condensedDescriptorOrigins.size(), condensedDescriptorOrigins.data()};

        if (DescriptorMethod::usesPointCloudInput()) {
            if(DescriptorMethod::hasGPUKernels()) {
                ShapeDescriptor::gpu::PointCloud gpuCloud = ShapeDescriptor::copyToGPU(pointCloud);
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuOrigins;
                if(potentiallyCondensedDescriptorOrigins.length > 0) {
                    gpuOrigins = ShapeDescriptor::copyToGPU(potentiallyCondensedDescriptorOrigins);
                }

                ShapeDescriptor::gpu::array<DescriptorType> gpuDescriptors = DescriptorMethod::computeDescriptors(gpuCloud, gpuOrigins, config, potentiallyCondensedSupportRadii, randomSeed);

                if(potentiallyCondensedDescriptorOrigins.length > 0) {
                    descriptors = ShapeDescriptor::copyToCPU(gpuDescriptors);
                    ShapeDescriptor::free(gpuDescriptors);
                    ShapeDescriptor::free(gpuOrigins);
                } else {
                    descriptors = ShapeDescriptor::cpu::array<DescriptorType>(0);
                }
                ShapeDescriptor::free(gpuCloud);
            } else {
                descriptors = DescriptorMethod::computeDescriptors(pointCloud, potentiallyCondensedDescriptorOrigins, config, potentiallyCondensedSupportRadii, randomSeed);
            }
        } else {
            if(DescriptorMethod::hasGPUKernels()) {
                ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copyToGPU(mesh);
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuOrigins;
                if(potentiallyCondensedDescriptorOrigins.length > 0) {
                    gpuOrigins = ShapeDescriptor::copyToGPU(potentiallyCondensedDescriptorOrigins);
                }

                ShapeDescriptor::gpu::array<DescriptorType> gpuDescriptors = DescriptorMethod::computeDescriptors(gpuMesh, gpuOrigins, config, potentiallyCondensedSupportRadii, randomSeed);

                if(potentiallyCondensedDescriptorOrigins.length > 0) {
                    descriptors = ShapeDescriptor::copyToCPU(gpuDescriptors);
                    ShapeDescriptor::free(gpuDescriptors);
                    ShapeDescriptor::free(gpuOrigins);
                } else {
                    descriptors = ShapeDescriptor::cpu::array<DescriptorType>(0);
                }
                ShapeDescriptor::free(gpuMesh);
            } else {
                descriptors = DescriptorMethod::computeDescriptors(mesh, potentiallyCondensedDescriptorOrigins, config, potentiallyCondensedSupportRadii, randomSeed);
            }
        }

        assert(descriptors.length == potentiallyCondensedDescriptorOrigins.length);

        if(allDescriptorsActive) {
            std::copy(descriptors.content, descriptors.content + descriptors.length, outputDescriptors.begin());
        } else {
            // We have previously condensed the descriptors requested to be computed, so we need to expand them back here
            for(uint32_t i = 0; i < originalDescriptorIndices.size(); i++) {
                outputDescriptors.at(originalDescriptorIndices.at(i)) = descriptors.content[i];
            }
        }

        if(config.at("methodSettings").at(DescriptorMethod::getName()).contains("filterNaNsInDescriptors")
        && config.at("methodSettings").at(DescriptorMethod::getName()).at("filterNaNsInDescriptors")) {
            for (DescriptorType &descriptor: outputDescriptors) {
                for (int i = 0; i < (sizeof(descriptor) / sizeof(float)); i++) {
                    if (std::isnan(descriptor.contents[i])) {
                        descriptor.contents[i] = 0;
                    }
                }
            }
        }

        ShapeDescriptor::free(descriptors);
    }

    template<typename DescriptorMethod, typename DescriptorType>
    void computeDescriptors(
            const ShapeDescriptor::cpu::Mesh& mesh,
            const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
            const std::vector<bool>& whichDescriptorsToCompute,
            const nlohmann::json &config,
            const std::vector<float>& supportRadii,
            uint64_t pointCloudSamplingSeed,
            uint64_t descriptorRandomSeed,
            float pointCloudSamplePointScaleFactor,
            std::vector<DescriptorType>& outputDescriptors) {

        ShapeDescriptor::cpu::PointCloud pointCloud;
        if (DescriptorMethod::usesPointCloudInput()) {
            pointCloud = computePointCloud<DescriptorMethod>(mesh, config, pointCloudSamplePointScaleFactor, pointCloudSamplingSeed);

            /*DebugRenderer renderer;
            renderer.drawMesh(mesh);
            ShapeDescriptor::writeOBJ(mesh, "weirdmesh.obj");
            renderer.drawPointCloud(pointCloud, {0, 0, 1});
            std::cout << "Number of descriptors: " << descriptorOrigins.length << std::endl;
            for(int i = 0; i < descriptorOrigins.length; i++) {
                std::cout << descriptorOrigins.content[i].vertex << std::endl;
                float size = 0.05;
                renderer.drawLine(descriptorOrigins.content[i].vertex + ShapeDescriptor::cpu::float3{0, 0, -size}, descriptorOrigins.content[i].vertex + ShapeDescriptor::cpu::float3{0, 0, size}, whichDescriptorsToCompute.at(i) ? ShapeDescriptor::cpu::float3{0, 1, 0} : ShapeDescriptor::cpu::float3{1, 0, 0});
                renderer.drawLine(descriptorOrigins.content[i].vertex + ShapeDescriptor::cpu::float3{0, -size, 0}, descriptorOrigins.content[i].vertex + ShapeDescriptor::cpu::float3{0, size, 0}, whichDescriptorsToCompute.at(i) ? ShapeDescriptor::cpu::float3{0, 1, 0} : ShapeDescriptor::cpu::float3{1, 0, 0});
                renderer.drawLine(descriptorOrigins.content[i].vertex + ShapeDescriptor::cpu::float3{-size, 0, 0}, descriptorOrigins.content[i].vertex + ShapeDescriptor::cpu::float3{size, 0, 0}, whichDescriptorsToCompute.at(i) ? ShapeDescriptor::cpu::float3{0, 1, 0} : ShapeDescriptor::cpu::float3{1, 0, 0});
            }
            renderer.show("Testing");
            renderer.waitForClose();*/
        }

        computeDescriptors<DescriptorMethod, DescriptorType>(mesh, pointCloud, descriptorOrigins, whichDescriptorsToCompute, config, supportRadii, descriptorRandomSeed, outputDescriptors);

        if(DescriptorMethod::usesPointCloudInput()) {
            ShapeDescriptor::free(pointCloud);
        }
    }
}
