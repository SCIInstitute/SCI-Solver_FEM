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
#include <setup_solver.h>
#include <amg_level.h>
#include "cuda_resources.h"

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

int main(int argc, char** argv)
{
    // Set up signal handler and create killer thread
    signal(SIGINT, signalHandler);
    pthread_t MasterThread = pthread_self();
    pthread_t KillerThread;
    pthread_create(&KillerThread, NULL, WaitToKill, (void*) &MasterThread);
    bool verbose_output = false;

    AMG_Config cfg;
    Matrix_d A;
    TriMesh* meshPtr;
    TetMesh* tetmeshPtr;
    FEM2D* fem2d = new FEM2D;
    FEM3D* fem3d = new FEM3D;

    cfg.setParameter("cuda_device_num", 0);
    cfg.setParameter("algorithm", CLASSICAL);
    cfg.setParameter("max_threads_per_block", getMaxThreads(64,0));

    try {
        for (int i = 1; i < argc; i++) {
            if (strncmp(argv[i], "-matrixtri", 100) == 0 || strncmp(argv[i], "-mtri", 100) == 0) {
                cfg.setParameter("mesh_type", 0);
                // load a matrix stored in MatrixMarket format
                i++;
                string meshfile = string(argv[i]) + string(".ply");
                meshPtr = TriMesh::read(meshfile.c_str());

            } else if (strncmp(argv[i], "-matrixtet", 100) == 0 || strncmp(argv[i], "-mtet", 100) == 0) {
                cfg.setParameter("mesh_type", 1);
                // load a matrix stored in MatrixMarket format
                i++;
                string nodefile = string(argv[i]) + string(".node");
                string elefile = string(argv[i]) + string(".ele");
                tetmeshPtr = TetMesh::read(nodefile.c_str(), elefile.c_str());

            } else if (strncmp(argv[i], "-amg", 100) == 0) {
                i++;
                cfg.parseParameterString(argv[i]);
            } else if (strncmp(argv[i], "-c", 100) == 0) {
                i++;
                cfg.parseFile(argv[i]);
            }
        }

        Vector_d_CG b_d;
        Vector_d_CG x_d;
        setup_solver(cfg, meshPtr, tetmeshPtr, fem2d, fem3d,
        		     &A, &b_d, &x_d, verbose_output);

        if( verbose_output )
        {
            cusp::print(A);
            cusp::print(x_d);
            cusp::print(b_d);
        }
    }
    catch (const invalid_argument& e) {
    	cerr << "Invalid argument: " << e.what() << endl;
    	return 1;
    }
    catch (...) {
        throw;
    }

    delete fem2d;
    delete fem3d;
}
