#pragma once

#include <iostream>
#include <iomanip>
/****************************************************
 * Debugging tools
 ***************************************************/
template<class Vector>
void printVector(const char* label, const Vector &v)
{
  std::cout << label << ": ";
  for(int i=0;i<v.size();i++)
  {
    std::cout << std::setprecision(4) << std::setw(8) << v[i] << " ";    
  }
  std::cout << std::endl;
}

template<class Matrix>
void printDense(const Matrix& A)
{
  for(int i=0;i<A.num_rows;i++)
  {
    for(int j=0;j<A.num_cols;j++)
    {
      bool exists=false;
      for(int jj=A.row_offsets[i];jj<A.row_offsets[i+1];jj++)
      {
        if(j==A.column_indices[jj])
        {
          printf("%5.2f ",A.values[jj]);
          exists=true;
          break;
        }
      }
      if(!exists)
        printf("   X  ");
    }
    printf("\n");
  }
  printf("\n");
}

#include <fstream>
template<class Matrix>
void printMatrix(const Matrix& A, char* fname)
{
  std::ofstream fout;
  fout.open(fname);

  fout << "%%MatrixMarket matrix coordinate real general" << std::endl;
  fout << std::setprecision(16) << std::fixed << A.num_rows << " " << A.num_cols << " " << A.num_entries << std::endl;
  for(int i=0;i<A.num_rows;i++)
  {
    for (int j=A.row_offsets[i];j<A.row_offsets[i+1];j++)
    {
      int c=A.column_indices[j];
      typename Matrix::value_type v=A.values[j];
      fout << i << " " << c << " " << v << std::endl;
    }
  }
  fout.close();
}




