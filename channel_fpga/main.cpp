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

int level = 1;
int ciphersize = 0;
int N = 8192;
int32_t bitlength = 21;  // Bitlength of the plaintext
const uint64_t t = sci::default_prime_mod.at(bitlength);
BfvParameter param = BfvParameter::create_fpga_parameter(t);
BfvContext *pub_context_p, *context_p;
BfvContext context =
    BfvContext::create_random_context(param);

/* Global variable for comm. */
int party = 0;     // 1 for Alice(Server), 2 for Bob(Client)
int port = 32000;  // Port each party listens to
string address =
    "127.0.0.1";      // IP address of the other party
int num_threads = 4;  // Number of threads to use
int32_t sf = 7;  // The power of scale factor (i.e. scale
                 // factor : 2 ** sf)

int main(int argc, char **argv) {
  /***** Argument Parsing *****/

  ArgMapping amap;

  amap.arg(
      "r", party,
      "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
  amap.arg("port", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("nt", num_threads, "Number of Threads");
  amap.arg("ell", bitlength, "Uniform Bitwidth");
  amap.parse(argc, argv);

  /***** Crypto Info *****/
  BfvCiphertext ct_temp = context.new_ciphertext(1, level);
  ciphersize = ct_temp.serialize(param).size();

  /***** Setup Phase *****/
  StartComputation();
  /***** context transfer *****/

  context_p = &context;
  auto data = transfer_context(party, context_p);
  pub_context_p = &data;

  /***** Computation Phase *****/
  CryptoInfo info;

  if(party == sci::ALICE) {
    info = CryptoInfo(party, pub_context_p, level,
                      ciphersize, prime_mod, N);
  } else {
    info = CryptoInfo(party, context_p, level, ciphersize,
                      prime_mod, N);
  }

  int Ci = 1024; int Co = 64; int H = 16; int W = 32; int p = 0; int s = 1; int k = 1; int Ho = 9; int Wo = 9; int C = 64;
  Tensor<uint64_t> input(TensorShape(3, {Ci, H, W})); 
  Tensor<uint64_t> weight(TensorShape(4, {Co, Ci, k, k}));
  // for (int i = 0; i < Co * Ci * k * k; i++){
  //   weight.cached_data[i] = i / (Ci * k * k);
  // }
  // if (party == sci::ALICE){
  //   for (int i = 0; i < Ci * H * W ; i++){
  //     input.cached_data[i] = 0;
  //   }
  // }
  // else{
  //   for (int i = 0; i < Ci * H * W; i++){
  //     input.cached_data[i] = 1;
  //   }
  // }
  Conv2dChan<uint64_t> conv2d_chan(weight, Co, Ci, H, W, k, s, p, info, std::move(param));
  Tensor<uint64_t> output = conv2d_chan.forward(input);
  SSreconstruct<uint64_t>(output, party, prime_mod);
  // for (int i = 0; i < Co; i++){
  //   std::cout << i << std::endl;
  //   for (int j = 0; j < Ho; j++){
  //     for (int l = 0; l < Wo; l++){
  //       std::cout << (output.cached_data[i * Ho * Wo + j * Wo + l] + C / 2) / C << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
 
  EndComputation();
}