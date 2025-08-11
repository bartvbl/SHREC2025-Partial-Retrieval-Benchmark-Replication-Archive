

#include "PythonAdapter.h"

#include "pybind11/pybind11.h"
#include "fmt/format.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11_json.hpp>
#include <pybind11/numpy.h>
#include <iostream>
#include <barrier>
#include <queue>
#include <condition_variable>
#include <omp.h>


namespace py = pybind11;
using namespace pybind11::literals;

static std::string currentActivePythonMethod;
static py::object pythonScope;
static py::object adapterModule;
static py::object SimpleNamespace;

const std::string pythonSourceFileName = "method";
const std::filesystem::path pythonMethodsDirectory = "../scripts/pythonmethods";

namespace ShapeBench {
    struct PythonWorkQueueItem {
        explicit PythonWorkQueueItem(const nlohmann::json* _config, const std::vector<float>* _supportRadii)
            : config{_config}, supportRadii{_supportRadii} {}

        bool isTriangleMeshInput = false;
        ShapeDescriptor::cpu::Mesh mesh;
        ShapeDescriptor::cpu::PointCloud cloud;
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins;
        const nlohmann::json* config = nullptr;
        const std::vector<float>* supportRadii = nullptr;
        uint64_t randomSeed = 0;
        uint32_t entriesPerDescriptor = 0;
        uint32_t threadID = 0;
    };

    struct PythonWorkItemResult {
        ShapeDescriptor::cpu::array<float> descriptors;
        bool isReady = false;
        bool executionFailed = false;
        std::string errorTraceBack = "";
        std::string errorMessage = "";
    };

    std::mutex pythonWorkQueueLock;
    std::queue<PythonWorkQueueItem> pythonPendingWorkQueue;
    std::vector<PythonWorkItemResult> pythonFinishedWorkQueue;
    std::mutex waitingMutex;
    std::condition_variable waitingVariable;
    std::thread pythonThread;
    nlohmann::json currentActiveMethodMetadata;
    bool pythonCloseRequested = false;

    ShapeDescriptor::cpu::array<float> pythonAdapterThread_convertDescriptors(const py::array_t<float>& descriptors, size_t expectedNumberOfDescriptors, size_t expectedElementsPerDescriptor) {
        if(descriptors.ndim() != 2) {
            throw std::runtime_error("Python descriptor method did not return the right numer of dimensions. Expected: dimension 0 for descriptors, dimension 1 for contents of each descriptor.");
        }

        size_t descriptorCount = descriptors.shape()[0];
        uint32_t elementsPerDescriptor = descriptors.shape()[1];

        if(elementsPerDescriptor != expectedElementsPerDescriptor) {
            throw std::runtime_error(fmt::format("Python method returned {} elements per descriptor, "
                                                 "but the descriptor structure defined on the C++ side has {} elements. "
                                                 "These need to match. Please correct the one that is incorrect.",
                                                 elementsPerDescriptor, expectedElementsPerDescriptor));
        }

        if(descriptorCount != expectedNumberOfDescriptors) {
            throw std::runtime_error(fmt::format("Python method returned {} descriptors, but {} were requested.", descriptorCount, expectedNumberOfDescriptors));
        }

        ShapeDescriptor::cpu::array<float> convertedDescriptors(elementsPerDescriptor * expectedNumberOfDescriptors);
        size_t nextBufferIndex = 0;
        for(uint32_t descriptorIndex = 0; descriptorIndex < expectedNumberOfDescriptors; descriptorIndex++) {
            for(uint32_t elementIndex = 0; elementIndex < elementsPerDescriptor; elementIndex++) {
                convertedDescriptors.content[nextBufferIndex] = descriptors.at(descriptorIndex, elementIndex);
                nextBufferIndex++;
            }
        }

        return convertedDescriptors;
    }


    ShapeDescriptor::cpu::array<float> pythonAdapterThread_computeDescriptors(PythonWorkQueueItem& item) {
        size_t inputVertexCount = item.isTriangleMeshInput ? item.mesh.vertexCount : item.cloud.pointCount;
        ShapeDescriptor::cpu::float3* vertices = item.isTriangleMeshInput ? item.mesh.vertices : item.cloud.vertices;
        ShapeDescriptor::cpu::float3* normals = item.isTriangleMeshInput ? item.mesh.normals : item.cloud.normals;

        py::array_t<float> vertexArray({inputVertexCount, (size_t)3});
        py::array_t<float> normalArray({inputVertexCount, (size_t)3});
        py::array_t<float> descriptorOriginArray({item.descriptorOrigins.length, (size_t) 2, (size_t) 3});
        py::array_t<float> supportRadiusArray(item.supportRadii->size());

        for(size_t vertexIndex = 0; vertexIndex < inputVertexCount; vertexIndex++) {
            vertexArray.mutable_at(vertexIndex, 0) = vertices[vertexIndex].x;
            vertexArray.mutable_at(vertexIndex, 1) = vertices[vertexIndex].y;
            vertexArray.mutable_at(vertexIndex, 2) = vertices[vertexIndex].z;
        }

        for(size_t vertexIndex = 0; vertexIndex < inputVertexCount; vertexIndex++) {
            normalArray.mutable_at(vertexIndex, 0) = normals[vertexIndex].x;
            normalArray.mutable_at(vertexIndex, 1) = normals[vertexIndex].y;
            normalArray.mutable_at(vertexIndex, 2) = normals[vertexIndex].z;
        }

        for(size_t radiusIndex = 0; radiusIndex < item.supportRadii->size(); radiusIndex++) {
            supportRadiusArray.mutable_at(radiusIndex) = item.supportRadii->at(radiusIndex);
        }

        for(size_t originIndex = 0; originIndex < item.descriptorOrigins.length; originIndex++) {
            descriptorOriginArray.mutable_at(originIndex, 0, 0) = item.descriptorOrigins.content[originIndex].vertex.x;
            descriptorOriginArray.mutable_at(originIndex, 0, 1) = item.descriptorOrigins.content[originIndex].vertex.y;
            descriptorOriginArray.mutable_at(originIndex, 0, 2) = item.descriptorOrigins.content[originIndex].vertex.z;
            descriptorOriginArray.mutable_at(originIndex, 1, 0) = item.descriptorOrigins.content[originIndex].normal.x;
            descriptorOriginArray.mutable_at(originIndex, 1, 1) = item.descriptorOrigins.content[originIndex].normal.y;
            descriptorOriginArray.mutable_at(originIndex, 1, 2) = item.descriptorOrigins.content[originIndex].normal.z;
        }

        py::dict configurationDict = *item.config;

        py::object sceneGeometryObject = SimpleNamespace("vertices"_a=vertexArray, "normals"_a=normalArray);

        py::object computeDescriptors = adapterModule.attr(item.isTriangleMeshInput ? "computeMeshDescriptors" : "computePointCloudDescriptors");

        py::array_t<float> descriptors = computeDescriptors(sceneGeometryObject, descriptorOriginArray, configurationDict, supportRadiusArray, item.randomSeed);

        return pythonAdapterThread_convertDescriptors(descriptors, item.descriptorOrigins.length, item.entriesPerDescriptor);
    }

    void pythonThreadMain(const std::string &pythonMethodName, const nlohmann::json &config) {
        py::initialize_interpreter(true, 0, nullptr, false);

        // Acquire the global lock
        py::gil_scoped_acquire GIL;

        // Add python method to path
        py::module_ sys = py::module_::import("sys");
        py::list path = sys.attr("path");
        path.append((pythonMethodsDirectory / pythonMethodName).string());
        sys.attr("path") = path;
        pythonScope = py::module_::import("__main__").attr("__dict__");
        adapterModule = py::module_::import(pythonSourceFileName.c_str());
        SimpleNamespace = py::module_::import("types").attr("SimpleNamespace");

        // Call init() on the python method
        py::object initMethodFunction = adapterModule.attr("initMethod");
        initMethodFunction(config);

        currentActiveMethodMetadata = adapterModule.attr("getMethodMetadata")();

        while(!pythonCloseRequested) {
            bool hasWork = false;
            PythonWorkQueueItem itemToProcess{nullptr, nullptr};
            {
                std::unique_lock<std::mutex> queueLock{pythonWorkQueueLock};
                hasWork = !pythonPendingWorkQueue.empty();
                if(hasWork) {
                    itemToProcess = pythonPendingWorkQueue.front();
                    pythonPendingWorkQueue.pop();
                }
            }

            if(hasWork) {
                PythonWorkItemResult result;
                try {
                    ShapeDescriptor::cpu::array<float> computedDescriptors = pythonAdapterThread_computeDescriptors(itemToProcess);
                    result.descriptors = computedDescriptors;
                    result.errorTraceBack = "";
                    result.errorMessage = "";
                    result.executionFailed = false;
                    result.isReady = true;
                } catch (py::error_already_set &e) {
                    py::object formatTraceBack = py::module_::import("traceback").attr("format_tb");
                    result.errorTraceBack = py::str(formatTraceBack(e.trace()));
                    result.errorMessage = e.what();
                    result.executionFailed = true;
                    result.isReady = true;
                } catch(const std::runtime_error& e) {
                    py::object formatTraceBack = py::module_::import("traceback").attr("format_tb");
                    result.errorTraceBack = "(not applicable, error was not in Python script)";
                    result.errorMessage = e.what();
                    result.executionFailed = true;
                    result.isReady = true;
                }

                {
                    std::unique_lock<std::mutex> lock{waitingMutex};
                    //std::cout << fmt::format("Results for thread are written by python thread, execution succeeded: {}\n", !result.executionFailed);
                    pythonFinishedWorkQueue.at(itemToProcess.threadID) = result;
                }

            } else {
                // Don't overload the system, sleep for a little bit
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // Have threads look at
            waitingVariable.notify_all();
        }

        // Clean up
        py::object destroyMethodFunction = adapterModule.attr("destroyMethod");
        destroyMethodFunction();
        py::finalize_interpreter();
    }

    ShapeDescriptor::cpu::array<float> waitForJobCompletion() {
        bool resultsReady = false;
        ShapeDescriptor::cpu::array<float> producedDescriptors;
        while(!resultsReady) {
            std::unique_lock<std::mutex> jobLock{waitingMutex};
            waitingVariable.wait(jobLock);
            PythonWorkItemResult& result = pythonFinishedWorkQueue.at(omp_get_thread_num());
            if(result.isReady) {
                //std::cout << fmt::format("Results for thread {} are ready, execution succeded: {}\n", omp_get_thread_num(), !result.executionFailed);
                if(result.executionFailed) {
                    throw std::runtime_error("Python method threw an exception during its execution.\n    Error: " + std::string(result.errorMessage) + "\nTraceback:\n" + result.errorTraceBack);
                }
                producedDescriptors = result.descriptors;
                result.isReady = false;
                result.executionFailed = false;
                result.errorTraceBack = "";
                result.errorMessage = "";
                resultsReady = true;
            }
        }
        return producedDescriptors;
    }
}





void ShapeBench::internal::initPython(const std::string &pythonMethodName, const nlohmann::json &config) {
    if(!currentActivePythonMethod.empty()) {
        throw std::runtime_error("Failed to initialise Python method \"" + pythonMethodName + "\": another method is already initialised: \"" + currentActivePythonMethod + "\"");
    }

    currentActivePythonMethod = pythonMethodName;

    uint32_t numberOfOMPThreads = 0;
    #pragma omp parallel
    {
        numberOfOMPThreads = omp_get_num_threads();
    }
    PythonWorkItemResult emptyResult;
    emptyResult.isReady = false;
    emptyResult.executionFailed = false;
    emptyResult.errorTraceBack = "";
    emptyResult.errorMessage = "";
    pythonFinishedWorkQueue.resize(numberOfOMPThreads, emptyResult);

    pythonThread = std::thread(ShapeBench::pythonThreadMain, pythonMethodName, config);
}

void ShapeBench::internal::destroyPython() {
    if(currentActivePythonMethod.empty()) {
        throw std::runtime_error("Failed to destroy Python method: no method is active");
    }

    pythonCloseRequested = true;
    pythonThread.join();
}

nlohmann::json ShapeBench::internal::getPythonMetadata() {
    return currentActiveMethodMetadata;
}



ShapeDescriptor::cpu::array<float>
ShapeBench::internal::computePythonDescriptors(const ShapeDescriptor::cpu::Mesh &mesh,
                                               const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                                               const nlohmann::json &config,
                                               const std::vector<float> &supportRadii,
                                               uint64_t randomSeed,
                                               uint32_t entriesPerDescriptor) {
    if(descriptorOrigins.length == 0) {
        std::cout << "    Python adapter warning: number of requested descriptors was 0, ignoring descriptor generation" << std::endl;
        return {0, nullptr};
    }

    PythonWorkQueueItem item(&config, &supportRadii);
    item.isTriangleMeshInput = true;
    item.mesh = mesh;
    item.descriptorOrigins = descriptorOrigins;
    item.randomSeed = randomSeed;
    item.entriesPerDescriptor = entriesPerDescriptor;
    item.threadID = omp_get_thread_num();

    {
        std::unique_lock<std::mutex> lock{pythonWorkQueueLock};
        //std::cout << fmt::format("Thread {} is enqueueing a new work item\n", omp_get_thread_num());
        pythonPendingWorkQueue.push(item);
    }

    return waitForJobCompletion();
}

ShapeDescriptor::cpu::array<float>
ShapeBench::internal::computePythonDescriptors(const ShapeDescriptor::cpu::PointCloud &cloud,
                                               const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                                               const nlohmann::json &config,
                                               const std::vector<float> &supportRadii,
                                               uint64_t randomSeed,
                                               uint32_t entriesPerDescriptor) {
    if(descriptorOrigins.length == 0) {
        std::cout << "    Python adapter warning: number of requested descriptors was 0, ignoring descriptor generation" << std::endl;
        return {0, nullptr};
    }

    PythonWorkQueueItem item(&config, &supportRadii);
    item.isTriangleMeshInput = false;
    item.cloud = cloud;
    item.descriptorOrigins = descriptorOrigins;
    item.randomSeed = randomSeed;
    item.entriesPerDescriptor = entriesPerDescriptor;
    item.threadID = omp_get_thread_num();

    {
        std::unique_lock<std::mutex> lock{waitingMutex};
        PythonWorkItemResult emptyResult;
        emptyResult.isReady = false;
        emptyResult.executionFailed = false;
        emptyResult.errorTraceBack = "";
        emptyResult.errorMessage = "";
        pythonFinishedWorkQueue.at(omp_get_thread_num()) = emptyResult;
    }

    {
        std::unique_lock<std::mutex> lock{pythonWorkQueueLock};
        //std::cout << fmt::format("Thread {} is enqueueing a new work item\n", omp_get_thread_num());
        pythonPendingWorkQueue.push(item);
    }

    return waitForJobCompletion();
}
