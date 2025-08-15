#include "shapeDescriptor/shapeDescriptor.h"
#include "dataset/Dataset.h"
#include "benchmarkCore/common-procedures/meshLoader.h"
#include "utils/prettyprint.h"
#include "shapeDescriptor/descriptors/SHOTGenerator.h"
#include "benchmarkCore/common-procedures/pointCloudSampler.h"
#include <fmt/format.h>
#include <random>

struct DummyMethod {
    inline bool hasConfigValue(const nlohmann::json& config, std::string methodName, std::string configEntryName) {
        return false;
    }

    static std::string getName() {
        return "dummy";
    }
};

int main(int argc, char** argv) {
    if(argc == 1) {
        throw std::runtime_error("Usage: executiontimevariation [statue or sphere]");
    }
    std::string mode = argv[1];
    if(mode != "statue" && mode != "sphere") {
        throw std::runtime_error("Usage: executiontimevariation [statue or sphere]");
    }

    bool isSphereMode = mode == "sphere";

    const uint32_t randomSeed = 27062025;
    const uint32_t experimentCount = 100000;
    const uint32_t iterationCount = 25;
    const uint32_t pointCloudPointCount = 5000000;

    std::filesystem::path outputDirectory = "../output_paper/figure_5_execution_time_variation";
    std::filesystem::create_directories(outputDirectory);
    std::filesystem::path outputFile = outputDirectory / (isSphereMode ? "time_variation_sphere.csv" : "time_variation_statue.csv");
    std::string fileToLoad = isSphereMode ? "../input/sphere.obj" : "../input/20320c2290554cfca7655d03af495da4.cm";

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(fileToLoad);

    // Taken from the dataset.json file. Skips having to partially start up the entire benchmark
    ShapeBench::DatasetEntry entry;
    entry.computedObjectCentre = isSphereMode ? std::array<double,3>{0, 0, 0} : std::array<double,3>{-1.7085775569560964, -0.8829449695181175, 17.96578024203662};
    entry.computedObjectRadius = isSphereMode ? 1 : 18.2042543418707;

    ShapeBench::moveAndScaleMesh(mesh, entry);

    nlohmann::json emptyConfig;
    emptyConfig["methodSettings"]["dummy"] = {};
    ShapeDescriptor::cpu::PointCloud cloud = ShapeBench::computePointCloud<DummyMethod>(mesh, emptyConfig, 1.0f, randomSeed, pointCloudPointCount);

    std::minstd_rand0 engine{randomSeed};
    std::uniform_real_distribution<float> supportRadiusDistribution(0, 1.5);
    std::uniform_int_distribution<uint32_t> indexDistribution(0, mesh.vertexCount);

    std::ofstream outputStream(outputFile);
    outputStream << "Vertex index, Chosen support radius, Time to render " << iterationCount << " descriptors (s)" << std::endl;

    for(uint32_t experimentIndex = 0; experimentIndex < experimentCount; experimentIndex++) {
        uint32_t vertexIndex = indexDistribution(engine);
        ShapeDescriptor::OrientedPoint referencePoint = {mesh.vertices[vertexIndex], mesh.normals[vertexIndex]};
        std::vector<ShapeDescriptor::OrientedPoint> origins(iterationCount, referencePoint);
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> originArray {iterationCount, origins.data()};
        float supportRadius = supportRadiusDistribution(engine);
        std::vector<float> supportRadii(iterationCount, supportRadius);

        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

        ShapeDescriptor::cpu::array<ShapeDescriptor::SHOTDescriptor<>> descriptors = ShapeDescriptor::generateSHOTDescriptorsMultiRadius(cloud, originArray, supportRadii);

        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
        uint64_t nanosecondsTaken = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
        fmt::println("Time taken: {}", supportRadius, double(nanosecondsTaken) / double(1000000000UL));
        outputStream << fmt::format("{}, {}, {}\n", vertexIndex, supportRadius, double(nanosecondsTaken) / double(1000000000ULL)) << std::flush;

        ShapeDescriptor::free(descriptors);
    }

    ShapeDescriptor::free(cloud);
    ShapeDescriptor::free(mesh);
}
