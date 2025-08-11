#pragma once

#include <iostream>

namespace ShapeBench {
    inline void drawProgressBar(uint32_t completed, uint32_t total) {
        const int barSteps = 16;
        float progress = float(completed) / float(total);
        int stepsToDraw = int(barSteps * progress);
        std::cout << "[";
        for(int i = 0; i < stepsToDraw; i++) {
            std::cout << "=";
        }
        for(int i = 0; i < barSteps - stepsToDraw; i++) {
            std::cout << " ";
        }
        std::cout << "]";
    }

    inline std::string durationToString(std::chrono::duration<uint64_t, std::nano> duration) {
        std::stringstream outStream;
        uint64_t executionTimeMilliseconds = duration.count() / 1000000;
        const uint64_t millisecondsPerSecond = 1000;
        const uint64_t millisecondsPerMinute = 60 * 1000;
        const uint64_t millisecondsPerHour = 60 * 60 * 1000;
        const uint64_t millisecondsPerDay = 24 * 60 * 60 * 1000;
        uint64_t durationInDays = executionTimeMilliseconds / millisecondsPerDay;
        executionTimeMilliseconds -= durationInDays * millisecondsPerDay;
        uint64_t durationInHours = executionTimeMilliseconds / millisecondsPerHour;
        executionTimeMilliseconds -= durationInHours * millisecondsPerHour;
        uint64_t durationInMinutes = executionTimeMilliseconds / millisecondsPerMinute;
        executionTimeMilliseconds -= durationInMinutes * millisecondsPerMinute;
        uint64_t durationInSeconds = executionTimeMilliseconds / millisecondsPerSecond;

        if(durationInDays > 0) {
            outStream << durationInDays << "d";
        }
        if(durationInHours > 0) {
            outStream << (durationInHours < 10 ? "0" : "") << durationInHours << ":";
        }
        outStream << (durationInMinutes < 10 ? "0" : "") << durationInMinutes << ":";
        outStream << (durationInSeconds < 10 ? "0" : "") << durationInSeconds;
        return outStream.str();
    }

    inline void printETA(std::chrono::steady_clock::time_point startTime, uint64_t completedCount, uint64_t totalItemsToProcess) {
        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        std::chrono::duration<uint64_t, std::nano> elapsedTimeThusFar = currentTime - startTime;
        uint64_t elapsedTimeNanoseconds = elapsedTimeThusFar.count();
        double elapsedTimeSeconds = double(elapsedTimeNanoseconds) / 1000000000.0;
        double expectedTotalTimeSeconds = elapsedTimeSeconds * (double(totalItemsToProcess) / double(completedCount));
        std::chrono::duration<uint64_t, std::nano> expectedTotalTime
                = std::chrono::nanoseconds(uint64_t(expectedTotalTimeSeconds * 1000000000.0));
        std::cout << " - Time taken: " << ShapeBench::durationToString(elapsedTimeThusFar);
        std::cout << "/" << ShapeBench::durationToString(expectedTotalTime);
        std::cout << ", remaining: " << ShapeBench::durationToString(expectedTotalTime - elapsedTimeThusFar) << std::endl;
    }



}
