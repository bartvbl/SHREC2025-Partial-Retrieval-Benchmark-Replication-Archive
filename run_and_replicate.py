#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import random
import sys
import multiprocessing
import hashlib
import base64
from shutil import which
from scripts.simple_term_menu import TerminalMenu
from scripts.prettytable import PrettyTable

if not (sys.version_info.major == 3 and sys.version_info.minor >= 8):
    print("This script requires Python 3.8 or higher.")
    print("You are using Python {}.{}.".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

script_dir = os.path.dirname(__file__)
os.makedirs(os.path.join(script_dir, 'input/objaverse-cache'), exist_ok=True)
os.makedirs(os.path.join(script_dir, 'input/objaverse-uncompressed'), exist_ok=True)
os.makedirs(os.path.join(script_dir, 'input/download'), exist_ok=True)

python_environments = {
    "GEDI": {
        "directory": os.path.join(script_dir, 'env', 'conda-gedi'),
        "activation": "env/conda-gedi/bin/activate",
        "deactivation": "source deactivate",
        "binDir": "bin-conda-gedi"
    },
    "COPS": {
        "directory": os.path.join(script_dir, 'env', 'python-cops'),
        "activation": "env/python-cops/bin/activate",
        "deactivation": "deactivate",
        "binDir": "bin-python-cops"
    }
}
default_bin_dir = python_environments['COPS']['binDir']

def run_command_line_command(command, working_directory='.'):
    print('>> Executing command:', command)
    subprocess.run(command, shell=True, check=False, cwd=working_directory)

def run_command_line_command_in_python_env(command, python_environment, working_directory=None):
    print('>> Executing command:', command)
    global python_environments
    environment_meta = python_environments[python_environment]
    if working_directory is None:
        working_directory = environment_meta['binDir']
    activateCommand = 'source ' + os.path.relpath(os.path.join(script_dir, environment_meta['activation']), working_directory)
    env_command = '/bin/bash -ic \'' + activateCommand + ' && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && ' + command + ' && ' + environment_meta["deactivation"] + '\''
    print(' -> Wrapped environment command:', env_command)
    subprocess.run(env_command, shell=True, check=False, cwd=working_directory)

def ask_for_confirmation(message):
    confirmation_menu = TerminalMenu(["yes", "no"], title=message)
    choice = confirmation_menu.show()
    return choice == 0

def downloadFile(fileURL, tempFile, extractInDirectory, name, unzipCommand = 'p7zip -k -d {}'):
    if not os.path.isfile('input/download/' + tempFile) or ask_for_confirmation('It appears the ' + name + ' archive file has already been downloaded. Would you like to download it again?'):
        print('Downloading the ' + name + ' archive file..')
        run_command_line_command('wget --output-document ' + tempFile + ' ' + fileURL, 'input/download/')
    print()
    os.makedirs(extractInDirectory, exist_ok=True)
    if unzipCommand is not None:
        run_command_line_command(unzipCommand.format(os.path.join(os.path.relpath('input/download', extractInDirectory), tempFile)), extractInDirectory)
    #if ask_for_confirmation('Download and extraction complete. Would you like to delete the compressed archive to save disk space?'):
    #    os.remove('input/download/' + tempFile)
    print()

def downloadDatasetsMenu():
    download_menu = TerminalMenu([
        "Download all",
        "Download computed results (4.7GB download, 83.4GB uncompressed)",
        "Download cache files (4.0GB download, 4.4GB uncompressed)",
        'Download prebuilt conda environment (2.2GB download, 5.7GB uncompressed)',
        "back"], title='------------------ Download Datasets ------------------')

    while True:
        choice = download_menu.show() + 1

        if choice == 1 or choice == 2:
            downloadFile('https://ntnu.box.com/shared/static/ql21r340osh00dqy4atbju2u13ojt4vz.7z',
                         'precomputed_results.7z', 'precomputed_results/', 'Results computed by the authors')
        if choice == 1 or choice == 3:
            downloadFile('https://ntnu.box.com/shared/static/p13szk6gx60zfi55qwmw4mkbifkx460p.7z', 'cache.7z',
                         'cache', 'Precomputed cache files')
        if choice == 1 or choice == 4:
            downloadFile('https://ntnu.box.com/shared/static/b1jr4pmp0z7sbmwkvy5zsvqys0qzyf4g.gz', 'shapebench-gedi.tar.gz', python_environments["GEDI"]["directory"], 'prebuilt conda environment', 'tar -v -xzf {} -C .')
        if choice == 5:
            return

def installDependencies():
    dependencies_menu = TerminalMenu([
        "Install APT dependencies",
        "Install CUDA (note: should not be higher than version 12)",
        "Install Chromium (if you don't have Chrome available. Mandated by Kaleido, which generates charts)",
        "Install pip/conda dependencies",
        "back"], title='------------------ Install Dependencies ------------------')

    while True:
        choice = dependencies_menu.show() + 1

        if choice == 1:
            run_command_line_command('sudo apt install ninja-build cmake g++ git libwayland-dev libxkbcommon-x11-dev xorg-dev libssl-dev m4 texinfo libboost-dev libeigen3-dev wget xvfb python3-tk python3-pip libstdc++-12-dev libomp-dev python3-venv libglfw3-dev')
        if choice == 2:
            print()
            print('----------------------------------------------')
            print('To install CUDA, please go here:')
            print('https://developer.nvidia.com/cuda-downloads')
            print()
        if choice == 3:
            run_command_line_command('sudo apt install chromium')
        #if choice == 4:
        #    run_command_line_command('wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh --output-document Miniforge3-Linux-x86_64.sh', 'input/download/')
        #    run_command_line_command('/bin/bash Miniforge3-Linux-x86_64.sh', 'input/download/')
        if choice == 4:
            run_command_line_command('python3 bin/conda-unpack', python_environments["GEDI"]["directory"])
            run_command_line_command('chmod +x bin/deactivate', python_environments["GEDI"]["directory"])
            # Conda is being a pain
            run_command_line_command('cp include/crypt.h include/python3.8/crypt.h', python_environments["GEDI"]["directory"])
            #run_command_line_command_in_python_env('pip install pointnet2_ops_lib/', 'GEDI', os.path.join(script_dir, 'scripts', 'pythonmethods', 'GEDI', 'backbones'))
            if not os.path.exists('env/python-cops'):
                run_command_line_command('python3 -m venv env/python-cops')
            COPSBinDir = python_environments["GEDI"]["directory"]
            run_command_line_command_in_python_env('pip3 install numpy matplotlib plotly wcwidth kaleido', 'COPS', COPSBinDir)
            run_command_line_command_in_python_env('pip3 install torch_geometric', 'COPS', COPSBinDir)
            run_command_line_command_in_python_env('pip3 install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.6.0cu124', 'COPS', COPSBinDir)
            run_command_line_command_in_python_env('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124', 'COPS', COPSBinDir)
            run_command_line_command_in_python_env('pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html', 'COPS', COPSBinDir)
            print()
        if choice == 5:
            return

def compileProject():
    global python_environments
    if not os.path.exists(python_environments["GEDI"]["directory"]):
        print()
        print('ERROR: Compilation cannot proceed before the conda environment has been downloaded and installed.')
        print('Use the download and install dependencies menus to do so.')
        print()
        return
    if not os.path.exists(python_environments["COPS"]["directory"]):
        print()
        print('ERROR: Compilation cannot proceed before the python COPS environment has been installed.')
        print('Use the install dependencies menu to do so.')
        print()
        return




    cudaCompiler = ''
    if which('nvcc') is None:
        print()
        print('It appears that the CUDA NVCC compiler is not on your path.')
        print('This usually means that CMake doesn\'t manage to find it.')
        print('The most common path at which NVCC is found is: /usr/local/cuda/bin/nvcc')
        nvccPath = input('Please paste the path to NVCC here, write "default" to use the default path listed above, or leave empty to try and run CMake as-is: ')
        if nvccPath != '':
            if nvccPath == 'default':
                nvccPath = '/usr/local/cuda/bin/nvcc'
            cudaCompiler = ' -DCMAKE_CUDA_COMPILER=' + nvccPath

    run_command_line_command('./configure', 'lib/gmp-6.3.0/')
    run_command_line_command('make -j', 'lib/gmp-6.3.0/')


    for environmentName in python_environments:

        environment_meta = python_environments[environmentName]
        binPath = os.path.join(script_dir, environment_meta['binDir'])
        run_command_line_command('rm -rf ' + binPath)
        os.makedirs(binPath, exist_ok=True)
        print('--------------- COMPILING PROJECT FOR ENVIRONMENT {} ----------------'.format(environmentName))
        run_command_line_command_in_python_env('cmake ' + os.path.relpath(script_dir, binPath) + ' -DCMAKE_BUILD_TYPE=Release -G Ninja' + cudaCompiler, environmentName, binPath)
        if not os.path.isfile(os.path.join(binPath, 'build.ninja')):
            print()
            print('Failed to compile the project: CMake exited with an error.')
            print()
            return

        run_command_line_command_in_python_env('ninja', environmentName, binPath)

    print()
    print('Complete.')
    print()

def fileMD5(filePath):
    with open(filePath, 'rb') as inFile:
        return hashlib.md5(inFile.read()).hexdigest()


def generateReplicationSettingsString(node):
    if node['recomputeEntirely']:
        return 'recompute entirely'
    elif node['recomputeRandomSubset']:
        return 'recompute ' + str(node['randomSubsetSize']) + ' at random'
    else:
        return 'disabled'

def editSettings(node, name):
    download_menu = TerminalMenu([
        "Recompute entirely",
        "Recompute random subset",
        "Disable replication",
        "back"], title='------------------ Replication Settings for ' + name + ' ------------------')

    choice = download_menu.show() + 1

    if choice == 1:
        node['recomputeEntirely'] = True
        node['recomputeRandomSubset'] = False
    if choice == 2:
        node['recomputeEntirely'] = False
        node['recomputeRandomSubset'] = True
        print()
        numberOfSamplesToReplicate = int(input('Number of samples to replicate: '))
        node['randomSubsetSize'] = numberOfSamplesToReplicate
    if choice == 3:
        node['recomputeEntirely'] = False
        node['recomputeRandomSubset'] = False

    return node

def selectReplicationRandomSeed(originalSeed):
    download_menu = TerminalMenu([
        "Pick new seed at random",
        "Enter a specific random seed",
        "Keep previous seed (" + str(originalSeed) + ')̈́'], title='------------------ Select New Random Seed ------------------')

    choice = download_menu.show() + 1

    if choice == 1:
        return random.getrandbits(64)
    if choice == 2:
        selectedRandomSeed = input('Enter new random seed: ')
        return int(selectedRandomSeed)
    return originalSeed

def readConfigFile(path = 'cfg/config_replication.json'):
    with open(path, 'r') as cfgFile:
        config = json.load(cfgFile)
        return config

def writeConfigFile(config, path = 'cfg/config_replication.json'):
    with open(path, 'w') as cfgFile:
        json.dump(config, cfgFile, indent=4)

def generateThreadLimiterString(configEntry):
    if not 'threadLimit' in configEntry:
        return 'No thread limit'
    else:
        return 'limited to ' + str(configEntry['threadLimit']) + ' threads'

def applyThreadLimiter(config):
    print()
    print('Thread limits ensure the number of threads working on a particular filter does not exceed a set number')
    print('If you have problems with running out of memory, you can set a thread limiter, and see if that helps things')
    print('Since memory usage scales linearly with more threads, that can help your situation')
    print()
    while True:
        download_menu = TerminalMenu([
            'Clutter: ' + generateThreadLimiterString(config['experimentsToRun'][0]),
            'Occlusion: ' + generateThreadLimiterString(config['experimentsToRun'][1]),
            'Normal vector deviation: ' + generateThreadLimiterString(config['experimentsToRun'][2]),
            'Alternate triangulation: ' + generateThreadLimiterString(config['experimentsToRun'][3]),
            'Support radius deviation: ' + generateThreadLimiterString(config['experimentsToRun'][4]),
            'Alternate mesh resolution: ' + generateThreadLimiterString(config['experimentsToRun'][5]),
            'Gaussian noise: ' + generateThreadLimiterString(config['experimentsToRun'][6]),
            'Clutter and Occlusion: ' + generateThreadLimiterString(config['experimentsToRun'][7]),
            'Clutter and Gaussian noise: ' + generateThreadLimiterString(config['experimentsToRun'][8]),
            'Occlusion and Gaussian noise: ' + generateThreadLimiterString(config['experimentsToRun'][9]),
            "back"], title='------------------ Apply Thread Limiter ------------------')
        choice = download_menu.show()
        if choice < 10:
            print()
            print('Please enter the new thread limit. Use 0 to disable the limit.')
            limit = int(input('New thread limit: '))
            if limit == 0:
                if 'threadLimit' in config['experimentsToRun'][choice]:
                    tempCopy = config['experimentsToRun'][choice]
                    del tempCopy['threadLimit']
                    config['experimentsToRun'][choice] = tempCopy
            else:
                config['experimentsToRun'][choice]['threadLimit'] = limit
        else:
            return config

def changeReplicationSettings(config_file_to_edit):
    config = readConfigFile(config_file_to_edit)

    while True:
        download_menu = TerminalMenu([
            'Compute or replicate experimental results: ' + generateReplicationSettingsString(config['replicationOverrides']['experiment']),
            'Compute or replicate reference descriptor set: ' + generateReplicationSettingsString(config['replicationOverrides']['referenceDescriptorSet']),
            'Random seed used when selecting random subsets to replicate: ' + str(config['replicationOverrides']['replicationRandomSeed']),
            'Verify computed minimum bounding sphere of input objects: ' + ('enabled' if config['datasetSettings']['verifyFileIntegrity'] else 'disabled'),
            'Size of dataset file cache in GB: ' + str(config['datasetSettings']['cacheSizeLimitGB']),
            'Change location of dataset file cache: ' + config['datasetSettings']['compressedRootDir'],
            'Limit the number of threads per experiment (use this if you run out of RAM)',
            'Print individual experiment results as they are being generated: ' + ('enabled' if config['verboseOutput'] else 'disabled'),
            'Enable visualisations of generated occluded scenes and clutter simulations: ' + ('enabled' if config['filterSettings']['additiveNoise']['enableDebugCamera'] else 'disabled'),
            "back"], title='------------------ Configure Replication ------------------')

        choice = download_menu.show() + 1

        if choice == 1:
            config['replicationOverrides']['experiment'] = editSettings(config['replicationOverrides']['experiment'],'Benchmark Results')
        if choice == 2:
            config['replicationOverrides']['referenceDescriptorSet'] = editSettings(config['replicationOverrides']['referenceDescriptorSet'], 'Reference Descriptor Set')
        if choice == 3:
            config['replicationOverrides']['replicationRandomSeed'] = selectReplicationRandomSeed(config['replicationOverrides']['replicationRandomSeed'])
        if choice == 4:
            config['datasetSettings']['verifyFileIntegrity'] = not config['datasetSettings']['verifyFileIntegrity']
        if choice == 5:
            print()
            newSize = int(input('Size of dataset file cache in GB: '))
            config['datasetSettings']['cacheSizeLimitGB'] = newSize
            print()
        if choice == 6:
            print()
            chosenDirectory = input('Enter a directory path here. Write "choose" for a graphical file chooser: ')
            if chosenDirectory == "choose":
                from tkinter import filedialog
                from tkinter import Tk
                root = Tk()
                root.withdraw()
                chosenDirectory = filedialog.askdirectory()
            config['datasetSettings']['compressedRootDir'] = chosenDirectory
        if choice == 7:
            config = applyThreadLimiter(config)
        if choice == 8:
            config['verboseOutput'] = not config['verboseOutput']
        if choice == 9:
            config['filterSettings']['additiveNoise']['enableDebugCamera'] = not config['filterSettings']['additiveNoise']['enableDebugCamera']
            if config['filterSettings']['additiveNoise']['enableDebugCamera']:
                warningBox = TerminalMenu([
                    "Ok"], title='Note: enabling these visualisations will likely cause filters that rely on OpenGL rendering to not replicate properly.')

                warningBox.show()

        if choice == 10:
            writeConfigFile(config, config_file_to_edit)
            return

def generateRadiusReplicationSettingsString(config):
    if config['replicationOverrides']['supportRadius']['recomputeEntirely']:
        return 'recompute entirely'
    elif config['replicationOverrides']['supportRadius']['recomputeSingleRadius']:
        selectedRadiusIndex = config['replicationOverrides']['supportRadius']['radiusIndexToRecompute']
        radiusMinValue = config['parameterSelection']['supportRadius']['radiusSearchStart']
        radiusStepValue = config['parameterSelection']['supportRadius']['radiusSearchStep']
        selectedRadius = str(radiusMinValue + float(selectedRadiusIndex) * radiusStepValue)
        return 'recompute statistics for radius ' + selectedRadius + ' only'
    else:
        return 'nothing is replicated'

allMethods = ['QUICCI', 'RICI', 'SHOT', 'COPS', 'GEDI', 'SI', 'MICCI-Triangle', 'MICCI-PointCloud']
trackExperiments = [
    ('experiment1-level1-occlusion-only',                'Experiment 1: Single occlusion filter'),
    ('experiment2-level1-clutter-only',                  'Experiment 2: Single clutter filter'),
    ('experiment3-level1-gaussian-noise-only',           'Experiment 3: Single gaussian noise filter'),
    ('experiment4-level2-occlusion-and-gaussian-noise',  'Experiment 4: Occlusion and Gaussian noise filter'),
    ('experiment5-level2-occlusion-both',                'Experiment 5: Occlusion on both objects'),
    ('experiment6-level2-occlusion-fixed-gaussian-both', 'Experiment 6: Occlusion and fixed level of gaussian noise on both objects'),
    ('experiment7-level2-occlusion-both-clutter',        'Experiment 7: Occlusion on both objects, clutter on one object'),
    ('experiment8-level2-occlusion-noise-both-clutter',  'Experiment 8: Occlusion and fixed level of gaussian noise on both objects, clutter on one object'),
    ('experiment9-level3-ultimate-test',                 'Experiment 9: Occlusion, two clutter objects, fixed level gaussian noise, and vertex perturbation')
]

def runCharter():
    os.makedirs('output_paper/charts', exist_ok=True)
    run_command_line_command_in_python_env('python3 tools/charter/charter.py --output-dir=output_paper/charts --results-directory=precomputed_results', 'COPS', script_dir)
    print()
    print('Charts created. You can find them in the output_paper/charts directory.')
    print()

def replicateExperimentResults(figureIndex, config_file_to_edit):
    global python_environments
    config = readConfigFile(config_file_to_edit)
    while True:
        print()
        print('Current replication settings:')
        print('- Experimental results:', generateReplicationSettingsString(config['replicationOverrides']['experiment']))
        print('- Replication of reference descriptor set:', generateReplicationSettingsString(config['replicationOverrides']['referenceDescriptorSet']))
        print('- Replication of sample object unfiltered descriptor set:', generateReplicationSettingsString(config['replicationOverrides']['sampleDescriptorSet']))
        print()
        replication_menu = TerminalMenu([
            'Edit replication settings (shortcut to same option in main menu)']
            + ['Subfigure ({}): {}'.format(list('abcdefgh')[index], method) for index, method in enumerate(allMethods)] + [
            "back"],
            title='------------------ Replicate Figure {}: {} ------------------'.format(7 + figureIndex, trackExperiments[figureIndex][1]))

        choice = replication_menu.show() + 1

        if choice == 1:
            changeReplicationSettings(config_file_to_edit)

        if choice > 1 and choice < len(allMethods) + 2:
            methodIndex = choice - 2
            methodName = allMethods[methodIndex]
            envName = methodName if methodName in python_environments else 'COPS'

            precomputedResultsDir = os.path.join('precomputed_results', trackExperiments[figureIndex][0])
            resultFiles = [x for x in os.listdir(precomputedResultsDir) if methodName in x]
            if len(resultFiles) != 1:
                raise Exception('There should be exactly one result file for each method in the precomputed results directory. Found {}.'.format(len(resultFiles)))
            fileToReplicate = os.path.join(precomputedResultsDir, resultFiles[0])

            print()
            enableVisualisations = config['filterSettings']['additiveNoise']['enableDebugCamera']
            commandPreamble = 'xvfb-run ' if not enableVisualisations else ''
            run_command_line_command_in_python_env(commandPreamble + './shapebench --replicate-results-file=../{} --configuration-file=../{}'.format(fileToReplicate, config_file_to_edit), envName)
            print(commandPreamble + './shapebench --replicate-results-file=../{} --configuration-file=../{}'.format(fileToReplicate, config_file_to_edit))
            print()
            print('Complete.')
            print('If you enabled any replication options in the settings, these have been successfully replicated if you did not receive a message about it, or the program has exited with an exception.')
            print('A comparison of the replicated benchmark results should be in a table that is visible a little bit above this message (you may need to scroll a bit)')
            print()
        if choice == 1 + len(allMethods) + 1:
            return

def replicateExperimentsFigures(config_file_to_edit):
    figureNumbers = [
        '9', '10', '11', '12 and 13', '14', '15', '16 and 17', '18 and 19', '20'
    ]
    experiments_menu = TerminalMenu([
        "Edit replication settings (shortcut to same option in main menu)"] +
        ['Replicate {} (Figure {})'.format(x[1], figureNumbers[index]) for index, x in enumerate(trackExperiments)]
        + ['Generate charts from precomputed results',
           'back'],
        title='------------------ Replicate Benchmark Results ------------------')
    while True:

        choice = experiments_menu.show() + 1
        if choice == 1:  #
            changeReplicationSettings(config_file_to_edit)
        if choice > 1 and choice <= len(trackExperiments) + 1:
            if choice == len(trackExperiments) + 1:
                print()
                print('Note: this figure does not have any subfigures like the others do, but it was easier to reuse the same bit of code for replicating the results')
                print('The overview chart is based on all result sets combined, so you can replicate and verify those individually, just note that the menu lists non-existing subfigures.')
                print()
            replicateExperimentResults(choice - 2, config_file_to_edit)
        if choice == len(trackExperiments) + 2:  #
            runCharter()
        if choice == len(trackExperiments) + 3:  #
            return

def showExecutionTimesNotice():
    notice_menu = TerminalMenu(["Continue"],
                               title="-------------------------"
                                     "\nNote:"
                                     "\nYou are now about to run experiments measuring execution time."
                                     "\nTo get meaningful data, please enter your BIOS, and disable CPU frequency boosting."
                                     "\nRestart this script after you have done so."
                                     "\nYou will otherwise see a lot of variability in measured execution times.")
    notice_menu.show()


def runBenchmarkInExecutionTimeMode(config_file_to_edit, methodName, generateSampleMeshes = False):
    global python_environments
    config = readConfigFile(config_file_to_edit)
    config["executionTimeMeasurement"]["syntheticExperimentsSharedSettings"]["generateSampleMeshes"] = generateSampleMeshes
    config["executionTimeMeasurement"]["enabled"] = True
    config["methodSettings"][methodName]["enabled"] = True
    writeConfigFile(config, config_file_to_edit)
    envName = methodName if methodName in python_environments else 'COPS'

    if generateSampleMeshes:
        print()
        print("Generating sample synthetic execution time meshes..")
        destinationDirectory = config["executionTimeMeasurement"]["syntheticExperimentsSharedSettings"][
            "generatedMeshDirectory"]
        absoluteDestinationDirectory = os.path.abspath(os.path.join(python_environments[envName]['binDir'], destinationDirectory))
        os.makedirs(absoluteDestinationDirectory, exist_ok=True)
        print("You can find the generated meshes here:", absoluteDestinationDirectory)
        print()

    config = readConfigFile(config_file_to_edit)
    enableVisualisations = config['filterSettings']['additiveNoise']['enableDebugCamera']
    commandPreamble = 'xvfb-run ' if not enableVisualisations else ''
    commandToRun = 'taskset --cpu-list 3 ' + commandPreamble + './shapebench --configuration-file=../{}'.format(config_file_to_edit)
    envName = methodName if methodName in python_environments else 'COPS'
    run_command_line_command_in_python_env(commandToRun, envName)
    print()
    print('Complete.')

    config["executionTimeMeasurement"]["syntheticExperimentsSharedSettings"]["generateSampleMeshes"] = False
    config["executionTimeMeasurement"]["enabled"] = False
    config["methodSettings"][methodName]["enabled"] = False
    writeConfigFile(config, config_file_to_edit)

def replicateExecutionTimes(config_file_to_edit):
    showExecutionTimesNotice()

    methodMenu = TerminalMenu(['Replicate execution times for method {}'.format(x) for x in allMethods] + ['Run charter script (same as the figures for the main experiment)', "back"], title='------------------ Replicate Execution Times ------------------')
    while True:
        choice = methodMenu.show() + 1
        if choice == len(allMethods) + 1:
            runCharter()
            continue
        if choice == len(allMethods) + 2:
            return
        methodName = allMethods[choice - 1]
        runBenchmarkInExecutionTimeMode(config_file_to_edit, methodName, False)


def replicateExecutionTimeVariabilityCharts(config_file_to_edit):
    showExecutionTimesNotice()

    variantMenu = TerminalMenu(["Subfigure 5a: Sphere", "Subfigure 5b: Statue", "back"],
                               title="------------------ Replicate Figure 5 ------------------")
    while True:
        choice = variantMenu.show() + 1
        if choice != 3:
            print()
            print('Note: this tool will run for quite a while, and produces execution times that are not comparable directly.')
            print('By default, it will do 100000 iterations, which each take several seconds.')
            print('The trend this chart is trying to show should already start to appear earlier though.')
            count = int(input('How many samples do you want to compute a chart for? '))
            print()
            variant = 'sphere'
            if choice == 2:
                variant = 'statue'

            run_command_line_command("./executiontimevariation {} {}".format(variant, count), "bin")

            output_file = 'output_paper/figure_5_execution_time_variation/time_variation_{}.csv'.format(variant)
            chart_file = 'output_paper/figure_5_execution_time_variation/chart_time_variation_{}.pdf'.format(variant)

            print()
            print('Data computed.')
            print('The replicated data was written to: {}'.format(output_file))
            print('You can find the version computed by the Authors at: precomputed_results/figure-5-exeuction-time-variation/time_variation_{}.csv'.format(variant))
            print()
            print('Computing chart based on the produced measurements..')
            run_command_line_command('python3 scripts/plot_time_variation_heatmap.py {} {}'.format(output_file, chart_file), '.')
            print('Done.')
            print()
            print('The chart was written to: {}'.format(chart_file))
            print()
        elif choice == 3:
            return


def replicateSyntheticExecutionTimeMeshes(config_file_to_edit):
    runBenchmarkInExecutionTimeMode(config_file_to_edit, "SHOT", True)


def replicateMICIDensityThreshold(config_file_to_edit):
    global default_bin_dir
    run_command_line_command('./miccithresholdselector --configuration-file=../{}'.format(config_file_to_edit), default_bin_dir)

    config = readConfigFile(config_file_to_edit)

    print()
    print('Complete.')
    print('The above threshold should be equal to the one from the config file:', config["methodSettings"]["MICCI-PointCloud"]["levelThreshold"])
    print("Note: this value differs from the one reported in the paper,")
    print("because the threshold is scaled as a function of the support radius.")
    print("The one in the paper is based on a support radius of 0.5,")
    print("while this one was computed for a radius of", config["methodSettings"]["MICCI-PointCloud"]["forSupportRadius"])
    print()


def runReplication(config_file_to_edit):
    while True:
        menu = TerminalMenu([
            "1. Change replication settings",
            "2. Replicate Figure 5 - Variation in execution times",
            "3. Replicate Figure 6 - Generate synthetic execution time meshes",
            "4. Replicate MICI density threshold",
            "5. Replicate Figure 9 to 20 - Benchmark results for experiments",
            "6. Replicate Figure 21 and 22 - Execution times",
            "7. back"
        ], title='---------------------- Replication Menu ----------------------')

        choice = menu.show() + 1
        
        match choice:
            case 1:
                changeReplicationSettings(config_file_to_edit)
            case 2:
                replicateExecutionTimeVariabilityCharts(config_file_to_edit)
            case 3:
                replicateSyntheticExecutionTimeMeshes(config_file_to_edit)
            case 4:
                replicateMICIDensityThreshold(config_file_to_edit)
            case 5:
                replicateExperimentsFigures(config_file_to_edit)
            case 6:
                replicateExecutionTimes(config_file_to_edit)
            case 7:
                return



# Run the experiment

def selectMethodsToRun(config_file_to_edit):
    config = readConfigFile(config_file_to_edit) # As a default it reads the config_replication.json
    methodList = list(config['methodSettings'].keys())
    
    while True:
        method_menu = TerminalMenu([f'{index}. {method}: {"enabled" if config["methodSettings"][method]["enabled"] else "disabled"}' for index, method in zip(list('123456789'), methodList)] + 
                                   ['back'],
                                   title='-' * 7 + f' Chose the method ' + '-' * 7)
        
        choice = method_menu.show() + 1
        
        match choice:
            case 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9:
                config['methodSettings'][methodList[choice - 1]]['enabled'] = False if config['methodSettings'][methodList[choice - 1]]['enabled'] else True
            case 10:
                writeConfigFile(config, config_file_to_edit)
                return

def selectFilters(exp, fKey):
    filters = [
        ('additive-noise', 'Additive Noise'),
        ('subtractive-noise', 'Subtractive Noise'),
        ('normal-noise', 'Alternate Triangulation'),
        ('repeated-capture', 'Normal Vector Noise'),
        ('support-radius-deviation', 'Support Radius Deviation'),
        ('depth-camera-capture', 'Depth Camera Capture'),
        ('gaussian-noise', 'Gaussian Noise')
    ]
    addedFilters = []
    
    while True:
        filter_menu = TerminalMenu([f'{index + 1}. {fName[1]}' for index, fName in enumerate(filters)] + 
                                   ['continue'], title='-' * 7 + ' Select one or more filters ' + '-' * 7)
        
        choice = filter_menu.show() + 1
        
        if choice > 0 and choice <= len(filters):
            if fKey not in exp.keys():
                exp[fKey] = [{'type': filters[choice - 1][0]}]
                addedFilters = [filters[choice - 1][1]]
            else:
                if filters[choice - 1][1] not in addedFilters:
                    exp[fKey].append({'type': filters[choice - 1][0]})
                    addedFilters.append(filters[choice - 1][1])
                else:
                    continue            
        else:
            return exp

def createNewExperiment(config_file_to_edit):
    newExperiment = {
                    "enabled": True,
                }
    
    while True:
        create_menu = TerminalMenu([
            'Step 1: Enter a name for the experiment (without spaces)',
            'Step 2: Chose the filters for the scene',
            'Step 3: Chose the filters for the model',
            'save experiment',
            'cancel'
        ], title='-' * 5 + ' Create a custom experiment ' + '-' * 5)
        
        choice = create_menu.show() + 1
        
        match choice:
            case 1:
                experimentName = ''
                checkName = False
                while not checkName:
                    experimentName = input('Name: ')
                    checkName = True if ' ' not in experimentName else print('/' * 10 + ' INVALID NAME ' + '\\' * 10)

                newExperiment['name'] = experimentName
            case 2:
                newExperiment = selectFilters(newExperiment, 'filters')
            case 3:
                newExperiment = selectFilters(newExperiment, 'modelFilters')
            case 4:
                if 'name' not in newExperiment.keys():
                    print('Please insert a valid name fot the experiment')
                elif 'filters' not in newExperiment.keys() and 'modelFilters' not in newExperiment.keys():
                    print('Specify at least one filter')
                else:
                    config = readConfigFile(config_file_to_edit)
                    config['experimentsToRun'].append(newExperiment)
                    
                    writeConfigFile(config, config_file_to_edit)

                    #TODO pass this down as parameter
                    base_config_file = config_file_to_edit.replace('_run', '_base')
                    
                    configBase = readConfigFile(base_config_file)
                    newExperiment['enabled'] = False
                    configBase['experimentsToRun'].append(newExperiment)
                    writeConfigFile(configBase, base_config_file)
                    
                    return
            case 5:
                return

def listExperimets(config_file_to_edit, expList):
    config = readConfigFile(config_file_to_edit)
    allExperimentsName = [experiment['name'] for experiment in config['experimentsToRun']]
    
    while True:
        subMenu = []
        for index, exp in enumerate(expList):
            posExp = allExperimentsName.index(exp)
            subMenu.append(f"{index + 1}. {exp}: {'enabled' if config['experimentsToRun'][posExp]['enabled'] else 'disabled'}")
        
        menu = TerminalMenu(subMenu + 
                            ['back'], title='-' * 10 + ' Enable experiments ' + '-' * 10)
        
        choice = menu.show() + 1
        
        if choice > 0 and choice <= len(expList):
            expIndex = allExperimentsName.index(expList[choice - 1])
            config['experimentsToRun'][expIndex]['enabled'] = False if config['experimentsToRun'][expIndex]['enabled'] else True
        else:
            writeConfigFile(config, config_file_to_edit)
            return

def enableExperimentsToRun(config_file_to_edit):
    # check if there are custom experiments
    
    #trackExperiments = [experiment[0] for experiment in trackExperiments]
    #trackExperimentsName = [experiment[0] for experiment in trackExperiments]
    #defaultsExperiments = trackExperiments + trackExperimentsName

    while True:
        config = readConfigFile(config_file_to_edit)
        customExperiments = [experiment['name'] for experiment in config['experimentsToRun']]
        
        experiment_menu = TerminalMenu([
            'Experiments List',
            'Define a new experiment',
            'back'
        ], title='-' * 10 + ' Experiment selection ' + '-' * 10)
    
        choice = experiment_menu.show() + 1

        match choice:
            case 1:
                listExperimets(config_file_to_edit, customExperiments)
            case 2:
                createNewExperiment(config_file_to_edit) #DONE
            case 3:
                return

def saveToFile(config):
    saveFiles = [fileName for fileName in os.listdir('cfg') if 'run' in fileName]
    root = 'cfg'
    
    while True:
        menu = TerminalMenu([f"{index + 1}. {fileName}" for index, fileName in enumerate(saveFiles)] + ['Create new file'] + ['back'],
                            title='-' * 5 + ' Select a file to save the configuration' +  '-' * 5)
        
        choice = menu.show() + 1
        
        if choice > 0 and choice <= len(saveFiles):
            savePath = os.path.join(root, saveFiles[choice - 1])
            
            with open(savePath, 'w') as f:
                json.dump(config, f, indent=4)
            
            return
        elif choice == len(saveFiles) + 1:
            fileName = input('Insert the file name:\n')
            
            if '.json' not in fileName:
                fileName += '.json'
            
            if 'run' not in fileName:
                tmpFileName = fileName.split('.')
                fileName = tmpFileName[0] + '_run.' + tmpFileName[1]
            
            savePath = os.path.join(root, fileName)
            
            with open(savePath, 'w') as f:
                json.dump(config, f, indent=4)
            
            return
        
        else:
            return
            
def runExperiments(config_file_to_edit):

    while True:
        menu = TerminalMenu([
            'Run configuration',
            'Edit benchmark settings',
            'Enable or disable experiments',
            'Enable or disable methods to test',
            'Back'
        ], title='-' * 10 + ' Run the benchmark ' + '-' * 10)

        choice = menu.show() + 1

        match choice:
            case 1:
                config = readConfigFile(config_file_to_edit)
                enableVisualisations = config['filterSettings']['additiveNoise']['enableDebugCamera']
                commandPreamble = 'xvfb-run ' if not enableVisualisations else ''

                # I think this should be a separate feature, maybe have it be its own option in the menu
                # The configuration file needs to be saved anyway because otherwise the benchmark cannot run it
                #saveToFile(config)

                print('Now running...')
                run_command_line_command(
                    commandPreamble + './shapebench --configuration-file=../{}'.format(config_file_to_edit), default_bin_dir)

            case 2:
                changeReplicationSettings()  # DONE
            case 3:
                enableExperimentsToRun(config_file_to_edit)
            case 4:
                selectMethodsToRun(config_file_to_edit)  # DONE
            case 5:
                return
    
def runMainMenu(config_file_to_edit):
    config = readConfigFile(config_file_to_edit)
    intendedForReplication = config['intendedForReplication'] if 'intendedForReplication' in config else False
    runOption = '4. Replicate results and experiments' if intendedForReplication else '4. Run experiments'

    while True:
        main_menu = TerminalMenu([
            "1. Download Author computed results and cache files",
            "2. Install dependencies",
            "3. Compile project",
            runOption,
            "5. Exit"], title='---------------------- Main Menu ----------------------')

        choice = main_menu.show() + 1
        
        match choice:
            case 1:
                downloadDatasetsMenu()
            case 2:
                installDependencies()
            case 3:
                compileProject()
            case 4:
                if intendedForReplication:
                    runReplication(config_file_to_edit)
                else:
                    runExperiments(config_file_to_edit)
            case 5:
                return


def computeConfigFileDisplayName(config_directory, config_file_name):
    with open(os.path.join(config_directory, config_file_name), 'r') as f:
        fileContents = json.loads(f.read())
    return tuple((os.path.join(config_directory, config_file_name), config_file_name + (": " + fileContents['description'] if 'description' in fileContents else "")))

def runIntroSequence():
    print()
    print('Greetings!')
    print()
    print('This script is intended to assist with replicating figures from previous papers,')
    print('as well as running the benchmark to produce new ones.')
    print()
    print('If you have not run this script before, you should run step 1 to 3 in order.')
    print('More details can be found in the replication manual PDF file that accompanies this script.')
    print()
    print('The script automatically edits benchmark configuration files.')
    print('These are configuration files in the \'cfg\' directory whose name ends with _base.json')
    print('Please select the configuration file which you would like to use for benchmark runs.')
    print()

    # Patching in absolute paths
    config = None
    
    loadConfig = True
    while loadConfig:
        configDirectory = 'cfg'
        allConfigs = [computeConfigFileDisplayName(configDirectory, fileName) for fileName in os.listdir(configDirectory) if '_base' in fileName]
        loadingMenu = TerminalMenu([f"{index + 1}. {fileName[1]}" for index, fileName in enumerate(allConfigs)],
                                   title='-' * 7 + ' Select the config file to use for this session ' +  '-' * 7)
        choice = loadingMenu.show() + 1

        config_file_to_edit = allConfigs[choice - 1][0]
        print('Selected config file:', config_file_to_edit)
        config = readConfigFile(config_file_to_edit)
        run_configuration_file = config_file_to_edit.replace('_base', '_run')
        print('Saving configuration to', run_configuration_file)
        print()
        if (os.path.isfile(run_configuration_file)):
            print('A configuration file for a previous run with this base configuration was found:')
            print('   ', run_configuration_file)
            print('Would you like to continue where you left off with this file?')
            print('It will otherwise be overwritten with the base configuration.')
            print()
            choice = ask_for_confirmation('Continue from previous run?')
            if choice == True:
                config = readConfigFile(run_configuration_file)
            
        writeConfigFile(config, run_configuration_file)

        loadConfig = False
    
    config['cacheDirectory'] = os.path.abspath(config['cacheDirectory'])
    config['resultsDirectory'] = os.path.abspath(config['resultsDirectory'])
    config['datasetSettings']['compressedRootDir'] = os.path.abspath(config['datasetSettings']['compressedRootDir'])
    config['datasetSettings']['objaverseRootDir'] = os.path.abspath(config['datasetSettings']['objaverseRootDir'])
    writeConfigFile(config)

    runMainMenu(run_configuration_file)

if __name__ == "__main__":

    runIntroSequence()
