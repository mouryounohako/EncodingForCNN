#include <vector>
#ifndef TENSOR_H
#define TENSOR_H

/*
 * The shape of a tensor.
 */
class TensorShape {
 public:
  int dim;
  std::vector<int> shape;

  TensorShape() : dim(0), shape({}) {}
  // Constructor with dimension and optional shape vector
  TensorShape(int dim, std::vector<int> shape = {})
      : dim(dim), shape(shape) {
    if(this->shape.empty()) {
      this->shape = std::vector<int>(
          dim, 0);  // Initialize with zeros if shape is not
                    // provided
    }
  }

  // Copy constructor
  TensorShape(const TensorShape& tensorshape)
      : dim(tensorshape.dim), shape(tensorshape.shape) {}

  ~TensorShape() {}
};

/*
 * Tensor object, contains tensor data and meta info.
 * @param Data: tensor data type, like uint64_t.
 */
template <typename Data>
class Tensor { 
 public:
  TensorShape tensorshape;
  std::vector<Data> cached_data;

  // Constructor with TensorShape and data vector
  Tensor() {
    tensorshape = TensorShape();
    cached_data = std::vector<Data>();
  }
  Tensor(const TensorShape& tensorshape,
         const std::vector<Data>& cached_data)
      : tensorshape(tensorshape),
        cached_data(cached_data) {}
  Tensor(const TensorShape& tensorshape)
      : tensorshape(tensorshape) {
    int size = 1;
    for(int i = 0; i < tensorshape.dim; i++) {
      size *= tensorshape.shape[i];
    }
    cached_data = std::vector<Data>(size, 0);
  }

  ~Tensor() {}

  void load(std::string pth) {
    if(pth == std::string("")){
      return;
    }
    std::ifstream file_weight(pth, std::ios::binary);
    int size = 1;
    for(int i = 0; i < tensorshape.dim; i++) {
      size *= tensorshape.shape[i];
    }
    file_weight.read(reinterpret_cast<char*>(cached_data.data()),
                     size * sizeof(Data));
  }
  int size() const {
    return cached_data.size();
  }
  Tensor<Data> operator+(const Tensor<Data>& other) {
    Tensor<Data> result = Tensor<Data>(tensorshape);
    assert(size() == other.size());
    for(int i = 0; i < other.size(); i++) {
      result.cached_data[i] = cached_data[i] + other.cached_data[i];
    }
    return result;
  }

  void operator+=(const Tensor<Data>& other) {
    for(int i = 0; i < size() ; i++) {
      cached_data[i] += other.cached_data[i];
    }
  }
  void operator%=(int prime_mod) {
    for(int i = 0; i < size() ; i++) {
      cached_data[i] %= prime_mod;
    }
  }
};

#endif  // TENSOR_H