Instructions for building the shared library:

1. Browse to the src directory in the terminal and run CMake. This can be through the CMake GUI (cmake-gui in linux/OSX), or CMake curses interface (ccmake).
2. In the CMake GUI, set the source and build directories to this project's src/ and build/, respectively.
3. Configure CMake settings as necessary. Options:
    a. Specify the CUDA_TOOLKIT_ROOT_DIR, which should be set to the path of the CUDA installation.
    b. Set BUILD_FEM_SOLVER_EXAMPLES to ON if example code needs to be generated.
4. Click the Configure and then Generate buttons in the CMake GUI.
    a. Fix CMake errors as necessary.
    b. When there are no errors, the code is now ready for a build.
5. Exit the CMake GUI, change to the build directory, and type 'make' in order to build the shared library file in the core directory.
