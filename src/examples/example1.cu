#include <stdio.h>
#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <exception>

#include <amg_config.h>
#include <types.h>
#include <TriMesh.h>
#include <tetmesh.h>
#include <cutil.h>
#include <FEM/FEM2D.h>
#include <FEM/FEM3D.h>
#include <timer.h>
#include <amg.h>

/*

#include <cusp/io/matrix_market.h>
#include <cusp/print.h>
#include <cusp/gallery/poisson.h>
#include <fstream>


*/

using namespace std;

void signalHandler(int signum)
{
//    DataRecorder::Add("Execution Halted", "By signal handler");
    pthread_exit(NULL);
}

void* WaitToKill(void* threadId)
{
    sleep(60);
    pthread_kill(*((pthread_t*) threadId), SIGINT);
    pthread_exit(NULL);
}

void printUsageAndExit()
{
    std::cout << "Usage: ./amgsolve [-m matrix | -p x y z] [-amg \"variable1=value1 variable2=value2 ... variable3=value3\" -help ] [-c config_file]\n";
    std::cout << "     -help display the command options\n";
    std::cout << "     -m specify the matrix input file\n";
    std::cout << "     -p points x y z:  use a poisson matrix on regular grid of size x y z\n";
    std::cout << "     -c set the amg solver options from the configuration file\n";
    std::cout << "     -amg set the amg solver options.  Options include the following:\n";
    AMG_Config::printOptions();

    exit(0);
}

int main(int argc, char** argv)
{

    // Set up signal handler and create killer thread
    signal(SIGINT, signalHandler);
    pthread_t MasterThread = pthread_self();
    pthread_t KillerThread;
    pthread_create(&KillerThread, NULL, WaitToKill, (void*) &MasterThread);

    // Zhisong's code to run the solver
    try {

        srand48(0);
        AMG_Config cfg;

        Matrix_d A;
        TriMesh* meshPtr;
        TetMesh* tetmeshPtr;
        FEM2D fem2d;
        FEM3D fem3d;

        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        int device;
        for (device = 0; device < deviceCount; ++device) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            size_t totalMemory = deviceProp.totalGlobalMem;
            int totalMB = totalMemory / 1000000;
            printf("Device %d (%s) has compute capability %d.%d, %d regs per block, and %dMb global memory.\n",
                    device, deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.regsPerBlock, totalMB);
        }

        cudaSetDevice(0);
        if (cudaDeviceReset() != cudaSuccess)
            exit(0);
        else {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 1);
        }

        double Assemblestart, Assemblestop;
        double neednbstart, neednbstop;
        double prepAssemstart, prepAssemstop;
        double setupstart, setupstop;

        for (int i = 1; i < argc; i++) {
            if (strncmp(argv[i], "-help", 100) == 0)
                printUsageAndExit();
            else if (strncmp(argv[i], "-matrixtri", 100) == 0 || strncmp(argv[i], "-mtri", 100) == 0) {
                i++;
                // load a matrix stored in MatrixMarket format
                string meshfile = string(argv[i]) + string(".ply");
                meshPtr = TriMesh::read(meshfile.c_str());
                meshPtr->rescale(4.0);
                neednbstart = CLOCK();
                meshPtr->need_neighbors();
                meshPtr->need_meshquality();
                neednbstop = CLOCK();
                Matrix_ell_d_CG Aell_d;
                Vector_d_CG RHS(meshPtr->vertices.size(), 0.0);

                prepAssemstart = CLOCK();
                trimesh2ell<Matrix_ell_d_CG > (meshPtr, Aell_d);
                cudaThreadSynchronize();

                prepAssemstop = CLOCK();
                Assemblestart = CLOCK();
                fem2d = FEM2D(meshPtr);

                fem2d.assemble(meshPtr, Aell_d, RHS);


                cudaThreadSynchronize();
                Assemblestop = CLOCK();

                A = Aell_d;
                Aell_d.resize(0, 0, 0, 0);
            } else if (strncmp(argv[i], "-matrixtet", 100) == 0 || strncmp(argv[i], "-mtet", 100) == 0) {
                i++;

                // load a matrix stored in MatrixMarket format
                string nodefile = string(argv[i]) + string(".node");
                string elefile = string(argv[i]) + string(".ele");

                tetmeshPtr = TetMesh::read(nodefile.c_str(), elefile.c_str());
                tetmeshPtr->need_neighbors();
                tetmeshPtr->need_meshquality();
                tetmeshPtr->rescale(1.0);

                Matrix_ell_d_CG Aell_d;
                Matrix_ell_h_CG Aell_h;
                Vector_d_CG RHS(tetmeshPtr->vertices.size(), 0.0);

                prepAssemstart = CLOCK();
                tetmesh2ell<Matrix_ell_d_CG > (tetmeshPtr, Aell_d);
                cudaThreadSynchronize();
                prepAssemstop = CLOCK();

                fem3d = FEM3D(tetmeshPtr);
                Assemblestart = CLOCK();
                fem3d.assemble(tetmeshPtr, Aell_d, RHS, true);
                cudaThreadSynchronize();
                Assemblestop = CLOCK();
                //            cusp::print(Aell_d);
                A = Aell_d;
                Aell_d.resize(0, 0, 0, 0);
            } else if (strncmp(argv[i], "-amg", 100) == 0) {
                i++;
                cfg.parseParameterString(argv[i]);
            } else if (strncmp(argv[i], "-c", 100) == 0) {
                i++;
                cfg.parseFile(argv[i]);
            }
        }

        if (A.num_rows == 0) {
            printf("Error no matrix specified\n");
            printUsageAndExit();
        }
        Vector_h_CG b(A.num_rows, 1.0);
        Vector_h_CG x(A.num_rows, 0.0); //initial
        Vector_d_CG x_d = x;
        Vector_d_CG b_d = b;

        cfg.printAMGConfig();
        AMG<Matrix_h, Vector_h> amg(cfg);
        amg.setup(A, meshPtr, tetmeshPtr);
        amg.printGridStatistics();
        amg.solve(b_d, x_d);


    }
    catch (exception& e) {
    }
    catch (...) {
        throw;
    }
}
