#include <inttypes.h>
#include <stdio.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "module.h"
#include "tensor.h"
#include "utils.h"

using namespace cxx_sdk_v2;
using namespace std;

int main() {

  // int Ci = 1; int Co = 1; int H = 3; int W = 3; int p = 0; int s = 1; int k = 2; int Ho = 2; int Wo = 2;
  // Tensor<uint64_t> input(TensorShape(3, {Ci, H, W})); 
  // Tensor<uint64_t> weight(TensorShape(4, {Co, Ci, k, k}));
  // for (int i = 0; i < Co * Ci * k * k; i++){
  //   weight.cached_data[i] = i + 1;
  // }
  // for (int i = 0; i < Ci * H * W; i++){
  //   input.cached_data[i] = i + 1;
  // }

  // Conv2dPlain<uint64_t> conv2dplain(weight, 1, 1, 3, 3, 2, 1, 0);
  // Tensor<uint64_t> output = conv2dplain.forward(input);
  // for (int i = 0; i < Co; i++){
  //   std::cout << i << std::endl;
  //   for (int j = 0; j < Ho; j++){
  //     for (int l = 0; l < Wo; l++){
  //       std::cout << output.cached_data[i * Ho * Wo + j * Wo + l] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  int sf = 7;
  ResNet18Plain<uint64_t> resnet18_plain(
      vector<string>(20, ""));
  Tensor<uint64_t> input(TensorShape(1, {3, 224, 224}));
  Tensor<uint64_t> output =
      resnet18_plain.forward(input, sf);
  
}