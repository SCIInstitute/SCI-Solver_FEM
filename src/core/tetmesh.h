#ifndef TETMESH_H
#define TETMESH_H
/*
Szymon Rusinkiewicz
Princeton University

TriMesh.h
Class for triangle meshes.
 */

#define  LARGENUM  10000000.0
#define  SMALLNUM  0.00000001
#define  ONE       1 
#define  CURVATURE 2 
#define  NOISE     3
#define  SPEEDTYPE ONE
#define  M_PI      3.14159265358979323846

#include "Vec.h"
#include <math.h>
#include <vector>
#include <list>
using std::vector;



//#define MIN(a,b) ( (a)< (b) )?(a):(b)
//#define MAX(a,b) ((a)>(b))?(a):(b)

class TetMesh
{
  //protected:
  //	static bool read_helper(const char *filename, TetMesh *mesh);

public:
  // Types

  struct Tet
  {
    int v[4];

    Tet()
    {
    }

    Tet(const int &v0, const int &v1, const int &v2, const int &v3)
    {
      v[0] = v0;
      v[1] = v1;
      v[2] = v2;
      v[3] = v3;
    }

    Tet(const int *v_)
    {
      v[0] = v_[0];
      v[1] = v_[1];
      v[2] = v_[2];
      v[3] = v_[3];
    }

    int &operator[] (int i)
    {
      return v[i];
    }

    const int &operator[] (int i)const
    {
      return v[i];
    }

    operator const int * () const
    {
      return &(v[0]);
    }

    operator const int * ()
    {
      return &(v[0]);
    }

    operator int * ()
    {
      return &(v[0]);
    }

    int indexof(int v_) const
    {
      return (v[0] == v_) ? 0 :
              (v[1] == v_) ? 1 :
              (v[2] == v_) ? 2 :
              (v[3] == v_) ? 3 : -1;
    }
  };

  // The basics: vertices and faces
  vector<point> vertices;
  vector<Tet> tets;
	vector<int> matlabels;
  // Connectivity structures:
  //  For each vertex, all neighboring vertices
  vector< vector<int> > neighbors;
  //  For each vertex, all neighboring faces
  vector< vector<int> > adjacenttets;
  vector<Tet> across_face;

  vector<double> radiusInscribe;

  void need_meshquality();

  void need_neighbors();
  void need_adjacenttets();
  void need_across_face();
  void need_meshinfo();
  void need_Rinscribe();
  void rescale(int size);

  static TetMesh *read(const char *nodefilename, const char* elefilename);
  //void write(const char *filename);

  // Debugging printout, controllable by a "verbose"ness parameter
  static int verbose;
  static void set_verbose(int);
  static int dprintf(const char *format, ...);

  //Constructor

  TetMesh()
  {
  }
};

#endif
