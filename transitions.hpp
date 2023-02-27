#ifndef TRANSITIONS_HPP
#define TRANSITIONS_HPP
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include "util.hpp"
#include "constants.h"

/* transition probabilities matrix class
   with size of Ns by Ns, stored in CSR
   format with Nnz non-zero values */
template <typename T>
class Transitions {
protected:
  unsigned int Ns;
  unsigned int Nnz;
private:
  unsigned int *col;
  unsigned int *row_ptr;
  T *data;

public:
  Transitions()
  {
    this->Ns = 0;
    this->Nnz = 0;

    col = NULL;
    row_ptr = NULL;
    data = NULL;
  }

  Transitions(unsigned int Ns_, unsigned int Nnz_)
  {
    this->Ns = Ns_;
    this->Nnz = Nnz_;

    col = new unsigned int[Nnz_];
    row_ptr = new unsigned int[Ns_+1];
    data = new T[Nnz_];
  }

  void SetCol(unsigned int i, unsigned int j)
  {
    col[j] = i;
  }

  void SetRowPtr(unsigned int p, unsigned int i)
  {
    row_ptr[i] = p;
  }

  void SetData(T value, unsigned int i)
  {
    data[i] = value;
  }

  void Eye()
  {
    row_ptr[0] = 0;

    for(unsigned int i = 0; i < this->Ns; i++)
    {
      row_ptr[i + 1] = row_ptr[i] + 1;
      col[i] = i;
      data[i] = 1;
    }
  }

  /* normalisation (columns) */
  void Norm()
  {
    T sum[this->Ns];
    memset(sum, 0.0, this->Ns*sizeof(T));

    for(unsigned int j = 0; j < this->Ns; j++)
      for(unsigned int i = row_ptr[j]; i < row_ptr[j+1]; i++)
        sum[col[i]] += data[i];

    for(unsigned int j = 0; j < this->Nnz; j++)
    {
      if (sum[col[j]] > 0)
        data[j] /= sum[col[j]];
      else
        data[j] /= this->Ns;
    }
  }

  unsigned int get_size()
  {
    return Ns;
  }

  unsigned int get_nnz()
  {
    return Nnz;
  }

  /* sparse matrix-vector multiplication */
  T *Txv(T *x)
  {
    T* y = new T[Ns];

    for(unsigned int j = 0; j < Ns; j++)
      y[j] = 0.0;

    for(unsigned int j = 0; j < Ns; j++)
    {
      T t = 0.0;

      for(unsigned int i = row_ptr[j]; i < row_ptr[j+1]; i++)
        t = t + data[i]*x[col[i]];

      y[j] = t;
    }

    return y;
  }

  /* sparse matrix-vector multiplication */
  void Txv(T *x, T *y)
  {
    T* _y = new T[Ns];

    for(unsigned int j = 0; j < Ns; j++)
      _y[j] = 0.0;

    for(unsigned int j = 0; j < Ns; j++)
      for(unsigned int i = row_ptr[j]; i < row_ptr[j+1]; i++)
        _y[j] = _y[j] + data[i]*x[col[i]];

    for(unsigned int j = 0; j < Ns; j++)
      y[j] = _y[j];

    delete [] _y;
  }

  void logTxv(T *x, std::vector<T> &y)
  {
    T _log_po=_log(p0);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(unsigned int i = 0; i < Ns; i++)
    {
      T t=0.;
      unsigned int itv = row_ptr[i];
      unsigned int itv_ = row_ptr[i+1];
      bool row_completed = false;

      for(unsigned int j = 0; j < Ns; j++)
      {
        if (itv == itv_)
	  row_completed = true;

        if (j == col[itv] && !row_completed)
	{
          //std::cout << "i=" << i << " j=" << j << " itv=" << itv << " data[" << itv << "]=" << data[itv] << " x[" << j << "]=" << x[j] << std::endl;
          t = t + _log(data[itv]+p0)*x[j];
	  itv++;
	}
	else
	{
          //std::cout << "i=" << i << " j=" << j << " itv=" << itv << " x[" << j << "]=" << x[j] << std::endl;
          t = t + _log_po*x[j];	
	}
      }

      y[i] = y[i] + t;
    }
  }

  /* store the f-th column in vector s */
  void extract_column(unsigned int f, std::vector<T> &s)
  {
    for(unsigned int i = 0; i < Ns; i++)
      for(unsigned int j = row_ptr[i]; j < row_ptr[i+1]; j++)
        if (col[j] == f)
	{
          s.at(i) = data[j];
          //std::cout << "extract_column: s[" << i << "] changed in " << data[j] << std::endl;
	}
  }

  /* extract the index corresponding to the maximum value in the f-th column */
  int MaxIndex(unsigned int f)
  {
    T max = 0;
    int maxindex = -1;

    for(unsigned int i = 0; i < Ns; i++)
      for(unsigned int j = row_ptr[i]; j < row_ptr[i+1]; j++)
        if (col[j] == f)
	{
          if (data[j] > max)
          {
            max = data[j];
	    maxindex = i;
          }
	}

    return maxindex;
  }

  /* constructor by passing a matrix */
  Transitions(std::vector<std::vector<T>> const &matrix)
  {
    this->Ns = matrix[0].size();
    this->Nnz = 0;
    for (std::vector<T> row: matrix)
      this->Nnz += std::count_if(row.begin(), row.end(), [](T c){return c > 0;});

    col = new unsigned int[this->Nnz];
    row_ptr = new unsigned int[this->Ns+1];
    data = new T[this->Nnz];

    std::size_t k = 0;
    for (std::size_t i = 0; i < matrix.size(); ++i)
    {
      bool first_found = false;
      for (std::size_t j = 0; j < matrix[0].size(); ++j)
      {
        if (matrix[i][j] != 0.0)
        {
          col[k] = j;
          if (!first_found)
          {
            row_ptr[i] = k;
            first_found = true;
          }
          data[k] = matrix[i][j];
          k++;
        }
      }
      if (!first_found)
        row_ptr[i] = k;
    }

    row_ptr[this->Ns] = k;
    //std::cout << "row_ptr[" << this->Ns << "]=" << row_ptr[this->Ns] << std::endl;
  }

  /* copy constructor by passing the object */
  Transitions(const Transitions<T> &t)
  {
    this->Ns = t.Ns;
    this->Nnz = t.Nnz;

    col = new unsigned int[this->Nnz];
    row_ptr = new unsigned int[this->Ns+1];
    data = new T[this->Nnz];

    for(unsigned int i = 0; i < this->Nnz; i++)
    {
      col[i] = t.col[i];
      data[i] = t.data[i];
    }

    for(unsigned int i = 0; i < this->Ns; i++)
      row_ptr[i] = t.row_ptr[i];

    row_ptr[this->Ns] = t.row_ptr[this->Ns];
  }

  ~Transitions() {
    delete [] col;
    delete [] row_ptr;
    delete [] data;
  }

  void Print()
  {
    for(unsigned int i = 0; i < Ns; i++)
      for(unsigned int j = row_ptr[i]; j < row_ptr[i+1]; j++)
        std::cout << "(" << i << "," << col[j] << ") " << data[j] << " ";
    std::cout << std::endl;
  }

  void csc_tocsr(unsigned int col_ptr[], unsigned int row[])
  {
    std::fill(this->row_ptr, this->row_ptr + this->Ns, 0);

    for (unsigned int n = 0; n < this->Ns; n++)
      this->row_ptr[row[n]]++;

    for(unsigned int j = 0, cumsum = 0; j < this->Ns; j++)
    {
      unsigned int temp  = this->row_ptr[j];
      this->row_ptr[j] = cumsum;
      cumsum += temp;
    }
    this->row_ptr[this->Ns] = this->Ns;

    for(unsigned int i = 0; i < this->Ns; i++)
      for(unsigned int jj = col_ptr[i]; jj < col_ptr[i+1]; jj++)
      {
        unsigned int j  = row[jj];
        unsigned int dest = this->row_ptr[j];

        this->col[dest] = i;

        this->row_ptr[j]++;
      }

    for(unsigned int j = 0, last = 0; j <= this->Ns; j++)
    {
      unsigned int temp = this->row_ptr[j];
      this->row_ptr[j] = last;
      last = temp;
    }
  }
};
#endif
