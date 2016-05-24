#include <cstdio>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string>
#include "tetmesh.h"

// Find the direct neighbors of each vertex
int TetMesh::verbose = 0;

void TetMesh::set_verbose(int v) { verbose = v; }

void TetMesh::need_meshquality()
{
  int max_valance = 0;
  int min_valance = 1000000;
  int sum = 0;
  int avg_valance = 0;
  FILE *valancefile, *reratiofile;
  valancefile = fopen("valance.txt", "w");
  reratiofile = fopen("reratio.txt", "w");


  for(int i = 0; i < neighbors.size(); i++)
  {
    int val = neighbors[i].size();
    sum += val;
    max_valance = max(max_valance, val);
    min_valance = min(min_valance, val);
    fprintf(valancefile, "%d\n", val);
  }
  avg_valance = sum / neighbors.size();
  if (verbose) {
    printf("Min valance is %d\n", min_valance);
    printf("Max valance is %d\n", max_valance);
    printf("average valance is %d\n", avg_valance);
  }
  int ne = tets.size();
  double mat0volume = 0.0;
  double totalvolume = 0.0;
  for(int i = 0; i < ne; i++)
  {
    //compute circum sphere radius
    Tet t = tets[i];
    point a = vertices[t[0]];
    point b = vertices[t[1]];
    point c = vertices[t[2]];
    point d = vertices[t[3]];
    point ab = b - a;
    point ac = c - a;
    point ad = d - a;

    point tmp1 = ab CROSS ac;
    point tmp2 = ad CROSS ab;
    point tmp3 = ac CROSS ad;
    point tmp4 = ac CROSS ad;

    double length1 = len2(ad);
    double length2 = len2(ac);
    double length3 = len2(ab);

    tmp1 = length1 * tmp1;
    tmp2 = length2 * tmp2;
    tmp3 = length3 * tmp3;

    point up = tmp1 + tmp2;
    up = up + tmp3;

    double u = len(up);
    double v = ab DOT tmp4;
    double volume = fabs(v)/6.0;
    totalvolume += volume;
    if(matlabels[i] == 0)
      mat0volume += volume;
    v = 2.0 * v;

    double radius = u / v;

    double edge1 = len(ab);
    double edge2 = len(ac);
    double edge3 = len(ad);
    double edge4 = len(b - c);
    double edge5 = len(c - d);
    double edge6 = len(d - b);

    double min_edge = (double)INT_MAX;
    min_edge = min(min_edge, edge1);
    min_edge = min(min_edge, edge2);
    min_edge = min(min_edge, edge3);
    min_edge = min(min_edge, edge4);
    min_edge = min(min_edge, edge5);
    min_edge = min(min_edge, edge6);

    double radius_edge = radius / min_edge;

    fprintf(reratiofile, "%.3f\n", radius_edge);

    //    printf("tet radius edge ratio is %f.2\n", radius_edge);

  }
  if (verbose) {
    printf("Total volume is %f\n", totalvolume);
    printf("Mat0 volume is %f\n", mat0volume);
    printf("Volume ratio is %f\n", mat0volume/totalvolume);
  }
  fclose(valancefile);
  fclose(reratiofile);
}

void TetMesh::need_neighbors()
{
  if(!neighbors.empty())
    return;

  if (verbose)
    std::cout << "Finding vertex neighbors... " << std::endl;
  int nv = vertices.size(), nt = tets.size();

  vector<int> numneighbors(nv, 0);
  for(int i = 0; i < nt; i++)
  {
    numneighbors[tets[i][0]]++;
    numneighbors[tets[i][1]]++;
    numneighbors[tets[i][2]]++;
    numneighbors[tets[i][3]]++;
  }


  neighbors.resize(nv);
  if( neighbors.size() != nv )
  {
    std::cerr << "Neighbors resize operation expected " << nv;
    std::cerr << " but returned " << neighbors.size() << std::endl;
  }

  int reserveSize = 0;
  for(int i = 0; i < nv; i++)
  {
    reserveSize = numneighbors[i] + 2;
    neighbors[i].reserve(reserveSize); // Slop for boundaries
    if( neighbors[i].capacity() != reserveSize )
    {
      std::cerr << "Failed to reserve neighbors[" << i << "]";
      std::cerr << "to size " << reserveSize;
      std::cerr << " (got " << neighbors[i].capacity() << ")." << std::endl;
    }
  }

  for(int i = 0; i < nt; i++)
  {
    for(int j = 0; j < 4; j++)
    {
      vector<int> &me = neighbors[tets[i][j]];
      int n1 = tets[i][(j + 1) % 4];
      int n2 = tets[i][(j + 2) % 4];
      int n3 = tets[i][(j + 3) % 4];

      if(find(me.begin(), me.end(), n1) == me.end())
        me.push_back(n1);
      if(find(me.begin(), me.end(), n2) == me.end())
        me.push_back(n2);
      if(find(me.begin(), me.end(), n3) == me.end())
        me.push_back(n3);
    }
  }
  if (verbose)
    std::cout << "Done.\n" << std::endl;
}

// Find the tets touching each vertex

void TetMesh::need_adjacenttets()
{
  if(!adjacenttets.empty())
    return;
  if (verbose)
    std::cout << "Finding adjacent tets... " << std::endl;
  int nv = vertices.size(), nt = tets.size();

  adjacenttets.resize(vertices.size());

  for(int i = 0; i < nt; i++)
  {
    for(int j = 0; j < 4; j++)
      adjacenttets[tets[i][j]].push_back(i);
  }

  int maxNumAjTets = 0;
  for(int i = 0; i < nv; i++)
  {
    maxNumAjTets = max(maxNumAjTets, (int)adjacenttets[i].size());

  }
  if (verbose) {
    printf("Max number of adjacent tet is: %d\n", maxNumAjTets);
    std::cout << "Done.\n" << std::endl;
  }
}

void TetMesh::need_across_face()
{
  if(!across_face.empty())
    return;
  need_adjacenttets();
  if (verbose)
    printf("Finding across-face maps... ");

  int nt = tets.size();
  across_face.resize(nt, Tet(-1, -1, -1, -1));

  for(int i = 0; i < nt; i++)
  {
    for(int j = 0; j < 4; j++)
    {
      if(across_face[i][j] != -1)
        continue;
      int v1 = tets[i][(j + 1) % 4];
      int v2 = tets[i][(j + 2) % 4];
      int v3 = tets[i][(j + 3) % 4];
      const vector<int> &a1 = adjacenttets[v1];
      const vector<int> &a2 = adjacenttets[v2];
      const vector<int> &a3 = adjacenttets[v3];
      for(int k1 = 0; k1 < a1.size(); k1++)
      {
        int other = a1[k1];
        if(other == i)
          continue;
        vector<int>::const_iterator it =
          find(a2.begin(), a2.end(), other);

        vector<int>::const_iterator it2 =
          find(a3.begin(), a3.end(), other);

        if(it == a2.end() || it2 == a3.end())
          continue;

        across_face[i][j] = other;
        break;



      }
    }
  }
  if (verbose)
    printf("Done.\n");
}

TetMesh *TetMesh::read(const char *nodefilename,
    const char* elefilename, const bool zero_based, const bool verb) {
  TetMesh *mesh = new TetMesh();
  mesh->set_verbose(verb);
  std::ifstream nodefile(nodefilename);
  std::ifstream elefile(elefilename);
  if(!nodefile.is_open() || !elefile.is_open())
  {
    printf("node or ele file open failed!\n");
    exit(0);
  }
  //get the number of nodes
  std::string line;
  int nv;
  int tmp;
  int labelIdx = 0;
  while (nodefile.good()) {
    std::getline(nodefile, line);
    if (line.empty() || line.at(0) == '#') continue;
    if (sscanf(line.c_str(), "%d %d %d %d", &nv, &tmp, &tmp, &tmp) != 4) {
      std::cerr << "Bad Node file" << std::endl;
      exit(0);
    }
    break;
  }
  mesh->vertices.resize(nv);
  int invalidNodeReferenceValue;
  if( zero_based )
	  invalidNodeReferenceValue = nv;
  else
	  invalidNodeReferenceValue = 0;

  //get the nodes
  size_t i = 0;
  int j = 0;
  while (nodefile.good()) {
    std::getline(nodefile, line);
    if (line.empty() || line.at(0) == '#') {
      continue;
    }
    float x, y, z;
    if (sscanf(line.c_str(), "%d %f %f %f", &tmp, &x, &y, &z) != 4) {
      std::cerr << "Bad Node file, line # " << i << std::endl;
      exit(0);
    }
    mesh->vertices[i][0] = x;
    mesh->vertices[i][1] = y;
    mesh->vertices[i][2] = z;
    i++;
  }
  nodefile.close();
  if (verbose) std::cout << "Read in node file successfully" << std::endl;
  //get the number of elements
  int ne;
  int haslabel;
  while (elefile.good()) {
    std::getline(elefile, line);
    if (line.empty() || line.at(0) == '#') continue;
    if (sscanf(line.c_str(), "%d %d %d", &ne, &tmp, &haslabel) != 3) {
      std::cerr << "Bad Ele file" << std::endl;
      exit(0);
    }
    break;
  }
  mesh->tets.resize(ne);
  mesh->matlabels.resize(ne, 0);
  mesh->materialValue.resize(haslabel, 0);

  i = 0;
  while (elefile.good())
  {
    std::getline(elefile, line);
    if (line.empty() || line.at(0) == '#') continue;
    int tval[4], mat;

    //Read from first part of the element file to get vertices comprising elements
    if( i < ne )
    {
      if (haslabel == 0)
      {
        if (sscanf(line.c_str(), "%d %d %d %d %d",
            &tmp, &tval[0], &tval[1], &tval[2], &tval[3]) != 5)
        {
          std::cerr << "Bad Ele file, line # " << i << std::endl;
          exit(0);
        }
      }
      else
      {
        if (sscanf(line.c_str(), "%d %d %d %d %d %d",
            &tmp, &tval[0], &tval[1], &tval[2], &tval[3], &mat) != 6)
        {
          std::cerr << "Bad Ele file, line # " << i << std::endl;
          exit(0);
        }
        mesh->matlabels[i] = mat;
      }

      for (j = 0; j < 4; ++j)
      {
        if( tval[j] == invalidNodeReferenceValue )
        {
          std::cerr << "Node reference error in elements file at element # " << i;
          std::cerr << ". Check if file is zero or one-based." << std::endl;
          exit(0);
        }
      }

      for (j = 0; j < 4; ++j)
        mesh->tets[i][j] = tval[j] - (zero_based?0:1);
    }
    //After elements section, read the material property values at the end of the file
    else if( haslabel > 0 )
    {
      if (sscanf(line.c_str(), "%f", &tmp) != 1)
      {
        std::cerr << "Bad material property in ele file, line # " << i << std::endl;
        exit(0);
      }
      if( labelIdx < haslabel )
        mesh->materialValue[labelIdx++] = tmp;
    }
    i++;
  }

  elefile.close();

  if (verbose) std::cout << "Read in ele file successfully" << std::endl;
  int minidx = INT_MAX;
  for(int i = 0; i < ne; i++)
  {
    for(int j = 0; j < 4; j++)
    {
      if(mesh->tets[i][j] < minidx)
        minidx = mesh->tets[i][j];
    }
  }
  if(minidx == 1)
  {
    if (verbose) printf("Minimum element index is 1\n");
    for(int i = 0; i < ne; i++)
    {
      for(int j = 0; j < 4; j++)
      {
        mesh->tets[i][j] = mesh->tets[i][j] - 1;
      }
    }

  }

  return mesh;

}

void TetMesh::rescale(int size)
{

  double minx = LARGENUM;
  double miny = LARGENUM;
  double minz = LARGENUM;
  double maxx = -LARGENUM;
  double maxy = -LARGENUM;
  double maxz = -LARGENUM;
  for(int v = 0; v < vertices.size(); v++)
  {
    double x = vertices[v][0];
    double y = vertices[v][1];
    double z = vertices[v][2];
    if(x < minx)
      minx = x;
    if(y < miny)
      miny = y;
    if(z < minz)
      minz = z;

    if(x > maxx)
      maxx = x;
    if(y > maxy)
      maxy = y;
    if(z > maxz)
      maxz = z;
  }
  for(int v = 0; v < vertices.size(); v++)
  {

    vertices[v][0] -= minx;
    vertices[v][1] -= miny;
    vertices[v][2] -= minz;


    vertices[v][0] = vertices[v][0] / (maxx - minx) * size;
    vertices[v][1] = vertices[v][1] / (maxy - miny) * size;
    vertices[v][2] = vertices[v][2] / (maxz - minz) * size;


  }
}


