/*
   Szymon Rusinkiewicz
   Princeton University

   TriMesh_connectivity.cc
   Manipulate data structures that describe connectivity between faces and verts.
 */


#include <stdio.h>
#include "TriMesh.h"
#include <algorithm>
using std::find;

void TriMesh::need_meshquality()
{
  int max_valance = 0;
  int sum = 0;
  int avg_valance = 0;
  FILE *valancefile, *reratiofile;
  valancefile = fopen("valance.txt", "w");
  reratiofile = fopen("reratio.txt", "w");


  for(int i =0; i<neighbors.size(); i++)
  {
    int val = neighbors[i].size();
    sum += val;
    max_valance = max(max_valance, val);
    fprintf(valancefile, "%d\n", val);
  }
  avg_valance = sum / neighbors.size();
  printf("Max valance is %d\n", max_valance);
  printf("average valance is %d\n", avg_valance);

  int ne = faces.size();
  for(int i =0; i<ne; i++)
  {
    //compute circum sphere radius
    Face t = faces[i];
    point A = vertices[t[0]];
    point B = vertices[t[1]];
    point C = vertices[t[2]];
    point ab = B - A;
    point ac = C - A;
    point bc = C - B;

    double a = len(bc);
    double b = len(ac);
    double c = len(ab);

    double radius = a*b*c / sqrt((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c));

    double min_edge = (double)INT_MAX;
    min_edge = min(min_edge, a);
    min_edge = min(min_edge, b);
    min_edge = min(min_edge, c);

    double radius_edge = radius / min_edge;

    fprintf(reratiofile, "%.3f\n", radius_edge);

  }

  fclose(valancefile);
  fclose(reratiofile);



}


void TriMesh::need_faceedges()
{
  if (faces.empty())
  {
    printf("No faces to compute face edges!!!\n");
    return;
  }
  int numFaces = faces.size();
  for (int i = 0; i < numFaces; i++)
  {
    Face f = faces[i];
    point edge01 = vertices[f[1]] - vertices[f[0]];
    point edge12 = vertices[f[2]] - vertices[f[1]];
    point edge20 = vertices[f[0]] - vertices[f[2]];
    faces[i].edgeLens[0] =sqrt(edge01[0]*edge01[0] + edge01[1]*edge01[1] + edge01[2]*edge01[2]);
    faces[i].edgeLens[1] =sqrt(edge12[0]*edge12[0] + edge12[1]*edge12[1] + edge12[2]*edge12[2]);
    faces[i].edgeLens[2] =sqrt(edge20[0]*edge20[0] + edge20[1]*edge20[1] + edge20[2]*edge20[2]);


  }

}

// Find the direct neighbors of each vertex
void TriMesh::need_neighbors()
{
  if (!neighbors.empty())
    return;

  printf("Finding vertex neighbors... ");
  int nv = vertices.size(), nf = faces.size();

  vector<int> numneighbors(nv);
  for (int i = 0; i < nf; i++) {
    numneighbors[faces[i][0]]++;
    numneighbors[faces[i][1]]++;
    numneighbors[faces[i][2]]++;
  }

  neighbors.resize(nv);
  for (int i = 0; i < nv; i++)
    neighbors[i].reserve(numneighbors[i]+2); // Slop for boundaries

  for (int i = 0; i < nf; i++) {
    for (int j = 0; j < 3; j++) {
      vector<int> &me = neighbors[faces[i][j]];
      int n1 = faces[i][(j+1)%3];
      int n2 = faces[i][(j+2)%3];
      if (find(me.begin(), me.end(), n1) == me.end())
        me.push_back(n1);
      if (find(me.begin(), me.end(), n2) == me.end())
        me.push_back(n2);
    }
  }

  printf("Done.\n");
}

void TriMesh::rescale(int size)
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
      miny= y;
    if(z < minz)
      minz = z;

    if(x> maxx)
      maxx = x;
    if(y> maxy)
      maxy = y;
    if(z> maxz)
      maxz = z;
  }
  for(int v = 0; v < vertices.size(); v++)
  {

    vertices[v][0] -= minx;
    vertices[v][1] -= miny;
    //vertices[v][2] -= minz;


    vertices[v][0] = vertices[v][0] / (maxx - minx) * size;
    vertices[v][1] = vertices[v][1] / (maxy - miny) * size;
    //vertices[v][2] = vertices[v][2] / (maxz - minz) * size;


  }
}

void TriMesh::meshoptimization(int iterNum)
{
  need_neighbors();
  for(int i=0; i<iterNum; i++)
  {
    for(int v = 0; v<vertices.size(); v++)
    {
      double x = vertices[v][0];
      double y = vertices[v][1];
      double z = vertices[v][2];
      double newx = 0.0, newy=0.0,newz=0.0;
      for(int b =0; b<neighbors[v].size(); b++)
      {
        newx += vertices[neighbors[v][b]][0];
        newy += vertices[neighbors[v][b]][1];
        newz += vertices[neighbors[v][b]][2];
      }
      newx /= neighbors[v].size();
      newy /= neighbors[v].size();
      newz /= neighbors[v].size();

      if(x != 0.0 && x != 16.0)
        vertices[v][0] = newx;

      if(y != 0.0 && y != 16.0)
        vertices[v][1] = newy;

      if(z != 0.0 && z != 16.0)
        vertices[v][2] = newz;
    }

  }

}
// Find the faces touching each vertex
void TriMesh::need_adjacentfaces()
{
  if (!adjacentfaces.empty())
    return;
  //  need_faces();

  printf("Finding vertex to triangle maps... ");
  int nv = vertices.size(), nf = faces.size();

  vector<int> numadjacentfaces(nv);
  for (int i = 0; i < nf; i++) {
    numadjacentfaces[faces[i][0]]++;
    numadjacentfaces[faces[i][1]]++;
    numadjacentfaces[faces[i][2]]++;
  }

  adjacentfaces.resize(vertices.size());
  for (int i = 0; i < nv; i++)
    adjacentfaces[i].reserve(numadjacentfaces[i]);

  for (int i = 0; i < nf; i++) {
    for (int j = 0; j < 3; j++)
      adjacentfaces[faces[i][j]].push_back(i);
  }

  printf("Done.\n");
}

// Find the face across each edge from each other face (-1 on boundary)
// If topology is bad, not necessarily what one would expect...
void TriMesh::need_across_edge()
{
  if (!across_edge.empty())
    return;
  need_adjacentfaces();

  printf("Finding across-edge maps... ");

  int nf = faces.size();
  across_edge.resize(nf, Face(-1,-1,-1));

  for (int i = 0; i < nf; i++) {
    for (int j = 0; j < 3; j++) {
      if (across_edge[i][j] != -1)
        continue;
      int v1 = faces[i][(j+1)%3];
      int v2 = faces[i][(j+2)%3];
      const vector<int> &a1 = adjacentfaces[v1];
      const vector<int> &a2 = adjacentfaces[v2];
      for (int k1 = 0; k1 < a1.size(); k1++) {
        int other = a1[k1];
        if (other == i)
          continue;
        vector<int>::const_iterator it =
          find(a2.begin(), a2.end(), other);
        if (it == a2.end())
          continue;
        int ind = (faces[other].indexof(v1)+1)%3;
        if (faces[other][(ind+1)%3] != v2)
          continue;
        across_edge[i][j] = other;
        across_edge[other][ind] = i;
        break;
      }
    }
  }

  printf("Done.\n");
}

