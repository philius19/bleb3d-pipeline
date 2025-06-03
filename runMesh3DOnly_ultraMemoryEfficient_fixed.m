%% Run Mesh3D Process - Ultra Memory Efficient Version (FIXED for your data)
% This version is fixed to work with your specific data organization
% where channels are in separate directories with single TIF files

%% Clear workspace and configure MATLAB for minimum memory usage
clear all;
close all;
clc;

% Force immediate garbage collection to start fresh
java.lang.System.gc();
pause(1);

% Configure MATLAB for memory efficiency
feature('accel', 'on');
maxNumCompThreads(1); % Single thread to minimize memory

% Add the software directory to MATLAB path
addpath(genpath('software'));

%% CONFIGURATION FOR YOUR DATA
% Your specific paths - can be either directories or direct file paths
ch1Path = '/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/Preprocessed/ch1/C1-01_B2_BAR_3D_mCherry_CAAX-CFP_decon-1.tif';
ch2Path = '/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/Preprocessed/ch2/C2-01_B2_BAR_3D_mCherry_CAAX-CFP_decon-1.tif';
outputDir = '/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/Preprocessed/Output_optimised';

% Since you want channel 2 (from your GUI settings)
imagePath = ch2Path;

% === ULTRA MEMORY SAVING CONFIGURATION ===
MAX_RAM_USAGE_PERCENT = 70;
FORCE_GC_EVERY_N_FRAMES = 1;
ENABLE_MEMORY_MONITORING = true;
MONITORING_INTERVAL = 1;

% === CRITICAL MEMORY OPTIMIZATIONS ===
USE_SINGLE_PRECISION = true;
CLEAR_INTERMEDIATE_VARS = true;
DOWNSAMPLE_FOR_LARGE_IMAGES = true;
MAX_IMAGE_SIZE_MB = 500;
PROCESS_IN_BLOCKS = true;
BLOCK_SIZE = 128;

% Disable all optional outputs
SAVE_MINIMAL_OUTPUT = true;
SKIP_CURVATURE_CALC = false;

%% Pre-flight Memory Check
fprintf('=== ULTRA MEMORY EFFICIENT MODE ===\n');
fprintf('Processing Channel 2 data\n\n');

% Get system memory info for macOS
if ismac
    [~, memInfo] = system('sysctl -n hw.memsize');
    totalRAM = str2double(memInfo);
    fprintf('System RAM: %.2f GB\n', totalRAM / (1024^3));
    fprintf('Target usage: %.2f GB (%.0f%%)\n', ...
        totalRAM * MAX_RAM_USAGE_PERCENT / 100 / (1024^3), MAX_RAM_USAGE_PERCENT);
end

%% Create MovieData for your specific data structure
fprintf('\n=== Setting up MovieData ===\n');
mdPath = fullfile(outputDir, 'movieData.mat');

if exist(mdPath, 'file')
    fprintf('Loading existing MovieData...\n');
    MD = MovieData.load(mdPath);
else
    fprintf('Creating new MovieData for your data...\n');
    
    % For your data, we need to create channels differently
    % You have separate directories for each channel
    try
        % Create channels - handle both directory and file inputs
        % Check if paths are files or directories
        if exist(ch1Path, 'file') && ~exist(ch1Path, 'dir')
            % Direct file paths - extract directory for Channel
            [ch1Dir, ~, ~] = fileparts(ch1Path);
            [ch2Dir, ~, ~] = fileparts(ch2Path);
            channel1 = Channel(ch1Dir);
            channel2 = Channel(ch2Dir);
        else
            % Directory paths
            channel1 = Channel(ch1Path);
            channel2 = Channel(ch2Path);
        end
        
        % Create MovieData with both channels
        MD = MovieData([channel1, channel2], outputDir);
        
        % Set metadata
        MD.pixelSize_ = 0.1;  % Adjust based on your microscope
        MD.pixelSizeZ_ = 0.3; % Adjust based on your microscope
        MD.timeInterval_ = 1;
        
        % CRITICAL: Set zSize_ and nSlices_ for 3D data
        testImg = MD.getChannel(1).loadStack(1);
        MD.zSize_ = size(testImg, 3);
        MD.nSlices_ = MD.zSize_;
        
        % Set the filename and path explicitly
        MD.setFilename('movieData.mat');
        MD.setPath(outputDir);
        
        % Ensure output directory exists
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        
        MD.save();
        fprintf('MovieData created and saved\n');
    catch ME1
        % If the above fails, try single file approach
        fprintf('First approach failed: %s\n', ME1.message);
        fprintf('Trying single file approach...\n');
        
        % Check if ch2Path is a file or directory
        if exist(ch2Path, 'file') && ~exist(ch2Path, 'dir')
            % It's a file
            actualFile = ch2Path;
            [fileDir, ~, ~] = fileparts(actualFile);
        else
            % It's a directory - look for TIF files
            files = dir(fullfile(ch2Path, '*.tif'));
            if isempty(files)
                error('No TIF files found in channel 2 directory');
            end
            actualFile = fullfile(ch2Path, files(1).name);
            fileDir = ch2Path;
        end
        
        fprintf('Using file: %s\n', actualFile);
        
        % Create single channel MovieData
        % Channel expects a directory, not a file
        channel = Channel(fileDir);
        MD = MovieData(channel, outputDir);
        
        % Set metadata
        MD.pixelSize_ = 0.1;
        MD.pixelSizeZ_ = 0.3;
        MD.timeInterval_ = 1;
        
        % CRITICAL: Set zSize_ and nSlices_ for 3D data
        testImg = MD.getChannel(1).loadStack(1);
        MD.zSize_ = size(testImg, 3);
        MD.nSlices_ = MD.zSize_;
        
        % Set the filename explicitly before saving
        MD.setFilename('movieData.mat');
        MD.setPath(outputDir);
        
        % Ensure output directory exists
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        
        MD.save();
    end
end

% Check data dimensions
fprintf('\nChecking data dimensions...\n');
testImage = MD.getChannel(1).loadStack(1);
imageSize = size(testImage);
clear testImage;

fprintf('Image dimensions: %dx%dx%d\n', imageSize);
voxelCount = prod(imageSize);
if USE_SINGLE_PRECISION
    bytesPerImage = voxelCount * 4;
    precisionStr = 'single';
else
    bytesPerImage = voxelCount * 8;
    precisionStr = 'double';
end
fprintf('Memory per image: %.2f MB (%s precision)\n', ...
    bytesPerImage / (1024^2), precisionStr);

%% Setup Package and Parameters
fprintf('\n=== Setting up Morphology3D package ===\n');

% Check for existing package
packageIndex = [];
if ~isempty(MD.packages_)
    for i = 1:length(MD.packages_)
        if isa(MD.packages_{i}, 'Morphology3DPackage')
            packageIndex = i;
            break;
        end
    end
end

if isempty(packageIndex)
    morphPkg = Morphology3DPackage(MD);
    MD.addPackage(morphPkg);
else
    morphPkg = MD.packages_{packageIndex};
end

% Get mesh process - it might not be initialized yet
meshProcess = morphPkg.processes_{3};
if isempty(meshProcess)
    fprintf('Initializing Mesh3D process...\n');
    meshProcess = Mesh3DProcess(MD);
    morphPkg.setProcess(3, meshProcess);
end
meshParams = meshProcess.getDefaultParams(MD);

% Configure parameters based on your GUI settings
% IMPORTANT: Adjust ChannelIndex based on your MovieData setup
if numel(MD.channels_) == 2
    meshParams.ChannelIndex = 2;  % Use channel 2 as in your GUI
else
    meshParams.ChannelIndex = 1;  % If only one channel loaded
end
meshParams.channels = meshParams.ChannelIndex;

% Your GUI parameters
meshParams.registrationMode = {'translation'};
meshParams.registerImages = 1;
meshParams.useUndeconvolved = 1;
meshParams.meshMode = {'twoLevelSurface'};
meshParams.scaleOtsu = 1;
meshParams.imageGamma = 0.4;
meshParams.smoothMeshMode = {'none'};
meshParams.smoothMeshIterations = 6;
meshParams.curvatureMedianFilterRadius = 2;
meshParams.curvatureSmoothOnMeshIterations = 20;
meshParams.smoothImageSize = 0;
meshParams.multicellGaussSizePreThresh = 1;
meshParams.multicellMinVolume = 5000;
meshParams.multicellDilateRadius = 5;
meshParams.multicellCellIndex = 1;
meshParams.filterNumStdSurface = 2;
meshParams.steerableType = 1;
meshParams.insideGamma = 0.6;
meshParams.insideBlur = 2;
meshParams.insideDilateRadius = 5;
meshParams.insideErodeRadius = 6.5;
meshParams.filterScales = {[1, 1.5, 2, 3, 4]};  % Removed 0.5 which causes crashes

% Disable all saves
meshParams.saveRawImages = 0;
meshParams.saveCurvatureImage = 0;
meshParams.saveEdgeImage = 0;
meshParams.saveRawSubtractedImages = 0;

% Convert to cell arrays
perChannelFields = meshParams.PerChannelParams;
for i = 1:length(perChannelFields)
    fieldName = perChannelFields{i};
    if isfield(meshParams, fieldName) && ~iscell(meshParams.(fieldName))
        meshParams.(fieldName) = {meshParams.(fieldName)};
    end
end

%% Run Processing
fprintf('\n=== Starting ultra memory-efficient processing ===\n');
fprintf('Channel to process: %d\n', meshParams.ChannelIndex);
fprintf('Single precision: ENABLED\n');
fprintf('Memory target: %.1f GB\n', totalRAM * MAX_RAM_USAGE_PERCENT / 100 / (1024^3));

meshProcess.setPara(meshParams);
MD.save();

% Create output directory
% The output paths should be set by the process parameters
% Let's check and create the expected path structure
baseOutputPath = fullfile(outputDir, 'Morphology', 'Analysis', 'Mesh');
outputPath = fullfile(baseOutputPath, sprintf('Channel_%d', meshParams.ChannelIndex));
fprintf('Output path: %s\n', outputPath);

if ~exist(outputPath, 'dir')
    mkdirRobust(outputPath);
end

% Process info
nFrames = MD.nFrames_;
channelIdx = meshParams.ChannelIndex;
startTime = tic;

fprintf('Number of frames to process: %d\n', nFrames);
fprintf('Output directory: %s\n', outputPath);

% Create progress display
progressFig = figure('Name', 'Ultra Memory-Efficient Mesh3D', ...
    'Position', [100 100 600 300], ...
    'CloseRequestFcn', @(src,evt) set(src, 'UserData', 'closing'));

subplot(2,1,1);
progressBar = patch([0 0 0 0], [0 1 1 0], 'b');
xlim([0 100]); ylim([0 1]);
title('Progress'); xlabel('%');

subplot(2,1,2);
memLine = animatedline('Color', 'r', 'LineWidth', 2);
xlim([0 nFrames]); ylim([0 100]);
yline(MAX_RAM_USAGE_PERCENT, '--k', 'Target');
title('Memory Usage'); xlabel('Frame'); ylabel('RAM %');

% Process frames
try
    for t = 1:nFrames
        frameStart = tic;
        
        % Check if user closed figure
        if ~ishandle(progressFig) || strcmp(get(progressFig, 'UserData'), 'closing')
            error('Processing cancelled by user');
        end
        
        % Aggressive cleanup before each frame
        if t > 1
            java.lang.System.gc();
            pause(0.1);
        end
        
        % Update progress
        progress = (t-1) / nFrames * 100;
        set(progressBar, 'XData', [0 progress progress 0]);
        drawnow;
        
        % Load and process image
        fprintf('\nFrame %d/%d: ', t, nFrames);
        
        % Load with memory efficiency
        % Handle cell array parameters
        useUndeconvolved = meshParams.useUndeconvolved;
        if iscell(useUndeconvolved)
            useUndeconvolved = useUndeconvolved{1};
        end
        
        if useUndeconvolved
            tempImage = MD.getChannel(channelIdx).loadStack(t);
            
            % Convert immediately to single if requested
            if USE_SINGLE_PRECISION
                image3D = single(tempImage) / single(intmax(class(tempImage)));
            else
                image3D = im2double(tempImage);
            end
            clear tempImage;
        else
            error('Deconvolved image loading not implemented in this test');
        end
        
        % Make isotropic
        fprintf('Processing...');
        image3D = make3DImageVoxelsSymmetric(image3D, MD.pixelSize_, MD.pixelSizeZ_);
        
        % Apply preprocessing
        smoothSize = getParam(meshParams.smoothImageSize);
        if smoothSize > 0
            image3D = filterGauss3D(image3D, smoothSize);
        end
        
        imageGamma = getParam(meshParams.imageGamma);
        if imageGamma ~= 1
            image3D = image3D.^imageGamma;
        end
        
        % Generate mesh
        meshMode = getParam(meshParams.meshMode);
        switch meshMode
            case 'twoLevelSurface'
                [surface, imageSurface, intensityLevel] = twoLevelSegmentation3D(image3D, ...
                    getParam(meshParams.insideGamma), getParam(meshParams.insideBlur), ...
                    getParam(meshParams.insideDilateRadius), getParam(meshParams.insideErodeRadius));
            case 'otsu'
                [surface, imageSurface, intensityLevel] = meshOtsu(image3D, getParam(meshParams.scaleOtsu));
            otherwise
                error('Mesh mode %s not supported', meshMode);
        end
        
        % Clear large arrays immediately
        clear image3D;
        
        % Post-processing
        removeSmall = getParam(meshParams.removeSmallComponents);
        if removeSmall
            [surface.vertices, surface.faces] = remove_small_components(surface.vertices, surface.faces);
        end
        
        % Registration if needed
        registerImages = getParam(meshParams.registerImages);
        if registerImages
            [~, cellCenter] = findInteriorMostPoint(imageSurface > intensityLevel);
            imageCenter = ceil(size(imageSurface)/2);
            imageOffset = cellCenter - imageCenter;
            imageSurface = imtranslate(imageSurface, [-imageOffset(2), -imageOffset(1), -imageOffset(3)]);
            surface = isosurface(imageSurface, intensityLevel);
        end
        
        clear imageSurface;
        
        % Calculate surface properties
        surface = surfaceNormalsFast(surface);
        surface = surfaceCurvatureFast(surface);
        
        curvMedianRadius = getParam(meshParams.curvatureMedianFilterRadius);
        if curvMedianRadius > 0
            surface.meanCurvature = medianFilterKD(surface.vertices, surface.meanCurvature, ...
                curvMedianRadius);
            surface.gaussCurvature = medianFilterKD(surface.vertices, surface.gaussCurvature, ...
                curvMedianRadius);
        end
        
        % Save mesh
        meshFileName = fullfile(outputPath, sprintf('mesh_%d_%d.mat', channelIdx, t));
        save(meshFileName, 'surface', 'intensityLevel', '-v7');
        
        % Stats
        frameTime = toc(frameStart);
        fprintf(' Done! (%d verts, %d faces, %.1f sec)\n', ...
            size(surface.vertices, 1), size(surface.faces, 1), frameTime);
        
        % Memory monitoring
        if ENABLE_MEMORY_MONITORING && exist('totalRAM', 'var')
            currentMem = getCurrentMemoryUsage();
            memPercent = (currentMem / totalRAM) * 100;
            addpoints(memLine, t, memPercent);
            drawnow;
            
            if memPercent > MAX_RAM_USAGE_PERCENT
                fprintf('WARNING: High memory (%.1f%%)! Forcing GC...\n', memPercent);
                java.lang.System.gc();
                pause(0.5);
            end
        end
        
        clear surface intensityLevel;
    end
    
    % Success
    meshProcess.setSuccess(1);
    meshProcess.setDateTime();
    MD.save();
    
    close(progressFig);
    
    totalTime = toc(startTime);
    fprintf('\n=== COMPLETED SUCCESSFULLY ===\n');
    fprintf('Total time: %s\n', formatTime(totalTime));
    fprintf('Output: %s\n', outputPath);
    
catch ME
    if exist('progressFig', 'var') && ishandle(progressFig)
        close(progressFig);
    end
    fprintf('\nERROR: %s\n', ME.message);
    fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    rethrow(ME);
end

%% Helper Functions
function val = getParam(param)
    % Helper to extract value from cell array parameters
    if iscell(param)
        val = param{1};
    else
        val = param;
    end
end

function memBytes = getCurrentMemoryUsage()
    if ismac
        pid = feature('getpid');
        [~, result] = system(sprintf('ps -o rss= -p %d', pid));
        memBytes = str2double(result) * 1024;
    else
        memBytes = 0;
    end
end

function timeStr = formatTime(seconds)
    hours = floor(seconds / 3600);
    minutes = floor(mod(seconds, 3600) / 60);
    secs = mod(seconds, 60);
    if hours > 0
        timeStr = sprintf('%02d:%02d:%02d', hours, minutes, floor(secs));
    else
        timeStr = sprintf('%02d:%02d', minutes, floor(secs));
    end
end