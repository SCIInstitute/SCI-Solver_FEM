#ifndef TETMESH_H
#define TETMESH_H
/*
   TetMesh: Class for tetrahedral meshes based on TriMesh by
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
#ifndef M_PI
#define  M_PI      3.14159265358979323846
#endif

#include "Vec.h"
#include <math.h>
#include <vector>
#include <list>

class TetMesh
{

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
  std::vector<point> vertices;
  std::vector<Tet> tets;
  std::vector<int> matlabels;
  // Connectivity structures:
  //  For each vertex, all neighboring vertices
  std::vector< std::vector<int> > neighbors;
  //  For each vertex, all neighboring faces
  std::vector< std::vector<int> > adjacenttets;
  std::vector<Tet> across_face;

  std::vector<double> radiusInscribe;

  void need_meshquality();

  void need_neighbors();
  void need_adjacenttets();
  void need_across_face();
  void need_meshinfo();
  void need_Rinscribe();
  void rescale(int size);

  //Tet mesh constructor
  //  nodefilename: file containing the XYZ position of each node or point.
  //    This must have the extension .node, and have the following
  //    characteristics: ASCII text with one node per line. Values are space-
  //    delimited. First line is a header line with 4 values: 'n 3 0 0'
  //    where n is the total number of nodes. Subsequent lines have the
  //    format 'i x y z' where i is the node number (starts at 1),
  //    and xyz are floats representing the node position in 3D space.
  //  elefilename: file containing the 4 nodes that define each tetrahedron.
  //    This must have the extension .ele, and have the following
  //    characteristics: ASCII text with one element per line. Values are
  //    space delimited. First line is a header line with 3 values: 't 4 0'
  //    where t is the total number of elements. Subsequent lines have the
  //    format 't a b c d' where t is the element number (starts at 1),
  //    and abcd are integers representing the node numbers from that file.
  //  zero_based: set to true if the element numbers in the file are zero-
  //    based (defaults to false).
  //  verbose: set to true for verbose output
  static TetMesh *read(const char *nodefilename, const char* elefilename, const bool verbose = false);
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
