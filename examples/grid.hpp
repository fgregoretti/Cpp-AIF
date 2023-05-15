#ifndef GRID_HPP
#define GRID_HPP
#include <assert.h>
#include <cmath>
#include <vector>
#include <cstdlib>

struct Coord {
  int x, y;

  Coord() : x(0), y(0) {
  }

  Coord(int _x, int _y): x(_x), y(_y) {
  }

  Coord operator*(int v) const {
    return Coord(this->x * v, this->y * v);
  }

  friend Coord& operator +=(Coord& left, const Coord& right) {
    left.x += right.x;
    left.y += right.y;
    return left;
  }

  friend const Coord operator +(const Coord& first, const Coord& second) {
    return Coord(first.x + second.x, first.y + second.y);
  }

  friend bool operator ==(const Coord& first, const Coord& second)
  {
    return first.x == second.x && first.y == second.y;
  }

  friend bool operator !=(const Coord& first, const Coord& second) {
    return first.x != second.x || first.y != second.y;
  }

  static double EuclideanDistance(Coord c1, Coord c2) {
   return sqrt((c1.x - c2.x) * (c1.x - c2.x) + (c1.y - c2.y) * (c1.y - c2.y));
  }
};

template<class T>
class Grid {
public:
  Grid() {
  }

  Grid(int dimx, int dimy) :
    dimx_(dimx),
    dimy_(dimy) {
    grid_.resize(dimx * dimy);
  }

  void Resize(int dimx, int dimy) {
    dimx_ = dimx;
    dimy_ = dimy;
    grid_.resize(dimx * dimy);
  }

  int dimx() const {
    return dimx_;
  }

  int dimy() const {
    return dimy_;
  }

  int Index(const Coord& coord) const {
    return coord.y * dimx_ + coord.x;
  }

  int Index(int x, int y) const {
    assert(Inside(x, y));
    return dimx_ * y + x;
  }

  bool Inside(const Coord& coord) const {
    return coord.x >= 0 && coord.y >= 0 && coord.x < dimx_
                        && coord.y < dimy_;
  }

  bool Inside(int x, int y) const {
    return x >= 0 && y >= 0
                  && x < dimx_
                  && y < dimy_;
  }

  T& operator()(const Coord& coord) {
    assert(Inside(coord));
    return grid_[Index(coord)];
  }

  const T& operator()(const Coord& coord) const {
    assert(Inside(coord));
    return grid_[Index(coord)];
  }

  T& operator()(int index) {
    assert(index < dimx_*dimy_);
    return grid_[index];
  }

  const T& operator()(int index) const {
    assert(index < dimx_*dimy_);
    return grid_[index];
  }

  T& operator()(int x, int y) {
    assert(Inside(x, y));
    return grid_[Index(x, y)];
  }

  const T& operator()(int x, int y) const {
    assert(Inside(x, y));
    return grid_[Index(x, y)];
  }

  void SetAllValues(const T& value) {
    for (int x = 0; x < dimx_; x++)
      for (int y = 0; y < dimy_; y++)
        grid_[Index(x, y)] = value;
  }

  T CoordToIndex(Coord c) const {
    return c.y * dimx() + c.x;
  }

  Coord IndexToCoord(int pos) const {
    return Coord(pos % dimx(), pos / dimx());
  }

  Coord GetCoord(int index) const {
    assert(index >= 0 && index < dimx_ * dimy_);
    return Coord(index % dimx_, index / dimx_);
  }

private:
  int dimx_, dimy_;
  std::vector<T> grid_;
};
#endif
