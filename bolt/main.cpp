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
BfvContext context = BfvContext::create_random_context(param);

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
  int d0 = 128; int d1 = 768; int d2 = 128;
  Tensor<uint64_t> input(TensorShape(2, {d0, d1})); 
  Tensor<uint64_t> weight(TensorShape(2, {d1, d2}));
  Tensor<uint64_t> output_ans(TensorShape(2, {d0, d2}));
  std::vector<uint64_t> tmp(d0 * d2, 0); 
  output_ans.cached_data = tmp;
  for (int i = 0; i < d2 * d1; i++){
    weight.cached_data[i] = 1;
  }
  if (party == sci::ALICE){
    for (int i = 0; i < d1 * d0; i++){
      input.cached_data[i] = 0;
    }
  }
  else{
    for (int i = 0; i < d1 * d0; i++){
      input.cached_data[i] = i;
    }
  }
  for (int i = 0; i < d0; i++){
    for (int j = 0; j < d1; j++){
      for (int k = 0; k < d2; k++){
        output_ans.cached_data[i * d2 + k] += input.cached_data[i * d1 + j] * weight.cached_data[j * d2 + k];
      }
    }
  }
  for (int i = 0; i < output_ans.size(); i++){
    output_ans.cached_data[i] = output_ans.cached_data[i] % prime_mod;
  }
  BoltMatmul<uint64_t> bolt_matmul(weight, d0, d1, d2, info, std::move(param));
  Tensor<uint64_t> output = bolt_matmul.forward(input);
  SSreconstruct<uint64_t>(output, party, prime_mod);
  std::cout << "Parity: " << (output.cached_data == output_ans.cached_data) << std::endl;

  EndComputation();
}