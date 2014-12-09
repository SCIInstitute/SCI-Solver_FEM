#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "tetmesh.h"

using namespace std;

// Find the direct neighbors of each vertex

void TetMesh::need_meshquality()
{
  int max_valance = 0;
  int sum = 0;
  int avg_valance = 0;
  FILE *valancefile, *reratiofile;
  valancefile = fopen("valance.txt", "w");
  reratiofile = fopen("reratio.txt", "w");


  for(int i = 0; i < neighbors.size(); i++)
  {
    int val = neighbors[i].size();
    sum += val;
    //		if(neighbors[i].size() > max_valance)
    //      max_valance = neighbors[i].size();
    max_valance = max(max_valance, val);
    fprintf(valancefile, "%d\n", val);
  }
  avg_valance = sum / neighbors.size();
  printf("Max valance is %d\n", max_valance);
  printf("average valance is %d\n", avg_valance);

  int ne = tets.size();
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

    //		printf("tet radius edge ratio is %f.2\n", radius_edge);

  }

  fclose(valancefile);
  fclose(reratiofile);



}

void TetMesh::need_neighbors()
{
  if(!neighbors.empty())
    return;


  cout << "Finding vertex neighbors... " << endl;
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
  for(int i = 0; i < nv; i++)
    neighbors[i].reserve(numneighbors[i] + 2); // Slop for boundaries

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

  cout << "Done.\n" << endl;
}


// Find the tets touching each vertex

void TetMesh::need_adjacenttets()
{
  if(!adjacenttets.empty())
    return;

  cout << "Finding adjacenttets... " << endl;
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

  printf("Max number of adjacent tet is: %d\n", maxNumAjTets);

  cout << "Done.\n" << endl;
}

void TetMesh::need_across_face()
{
  if(!across_face.empty())
    return;
  need_adjacenttets();

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

  printf("Done.\n");
}

TetMesh *TetMesh::read(const char *nodefilename, const char* elefilename)
{
  TetMesh *mesh = new TetMesh();
  FILE* nodefile = fopen(nodefilename, "r");
  FILE* elefile = fopen(elefilename, "r");
  if(nodefile == NULL || elefile == NULL)
  {
    printf("node or ele file open failed!");
    exit(0);
  }

  int nv;
  int tmp;
  fscanf(nodefile, "%d %d %d %d", &nv, &tmp, &tmp, &tmp);
  mesh->vertices.resize(nv);
  for(int i = 0; i < nv; i++)
  {
    float x, y, z;
    fscanf(nodefile, "%d %f %f %f", &tmp, &x, &y, &z);
    mesh->vertices[i][0] = x;
    mesh->vertices[i][1] = y;
    mesh->vertices[i][2] = z;
  }

  int ne;
  int haslabel;
  fscanf(elefile, "%d %d %d", &ne, &tmp, &haslabel);
  mesh->tets.resize(ne);
	mesh->matlabels.resize(ne, 0);

  if(haslabel == 0)
  {
    for(int i = 0; i < ne; i++)
    {
      fscanf(elefile, "%d %d %d %d %d", &tmp, &mesh->tets[i][0], &mesh->tets[i][1], &mesh->tets[i][2], &mesh->tets[i][3]);
    }
  }
  else
  {
		for(int i = 0; i < ne; i++)
    {
      fscanf(elefile, "%d %d %d %d %d", &tmp, &mesh->tets[i][0], &mesh->tets[i][1], &mesh->tets[i][2], &mesh->tets[i][3], &mesh->matlabels[i]);
    }
  }


	for(int i=0; i<ne; i++)
	{
		double x0, x1,x2,x3;
		x0 = mesh->vertices[mesh->tets[i][0]][0];
		x1 = mesh->vertices[mesh->tets[i][1]][0];
		x2 = mesh->vertices[mesh->tets[i][2]][0];
		x3 = mesh->vertices[mesh->tets[i][3]][0];
		
		double avg = (x0+x1+x2+x3) / 4.0;
		int temp = avg / 32.0;
		if(temp%2==0)
			mesh->matlabels[i] = 0;
		else
			mesh->matlabels[i] = 1;
	}

  //write new mesh //	FILE* brainnode;
  //	FILE* brainele;
  //
  //	brainnode = fopen("brain0.5m_new.node", "w+");
  //	brainele  = fopen("brain0.5m_new.ele", "w+");
  //
  //	fprintf(brainnode, "%d 3 0 0\n", &nv);
  //	double x,y,z;
  //  for(int i = 0; i < nv; i++)
  //  {
  //		x=mesh->vertices[i][0];
  //    y=mesh->vertices[i][1];
  //    z=mesh->vertices[i][2];
  //    fprintf(brainnode, "%d %.6f %.6f %.6f\n", i, x, y, z);
  //    
  //  }
  //
  //  fprintf(brainele, "%d 4 0\n", &ne);
  //  for(int i = 0; i < ne; i++)
  //  {
  //    fprintf(brainele, "%d %d %d %d %d\n", i, mesh->tets[i][0], mesh->tets[i][1], mesh->tets[i][2], mesh->tets[i][3]);
  //  }



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
    printf("Minimum element index is 1\n");
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


