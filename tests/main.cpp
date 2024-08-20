#include <inttypes.h>
#include <stdio.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "tensor.h"
#include "utils.h"

#include <random>
#include <memory>

#include "hexl/ntt/ntt.hpp"
#include "cxx_fhe_lib_v2.h"
#include "cxx_fpga_ops_v2.h"
#include "library_fixed.h"

using namespace cxx_sdk_v2;
using namespace std;

int level = 1;
int ciphersize = 0;
int N = 8192;
int32_t bitlength = 12;  // Bitlength of the plaintext
// const uint64_t t = sci::default_prime_mod.at(bitlength);
const uint64_t t = 0x1b4001;
// const uint64_t t = 1 << 21;
std::shared_ptr<BfvParameter> param = std::make_shared<BfvParameter>(BfvParameter::create_fpga_parameter(t));
BfvContext *pub_context_p, *context_p;
BfvContext context =
    BfvContext::create_random_context(*param);

/* Global variable for comm. */
int party = 0;     // 1 for Alice(Server), 2 for Bob(Client)
int port = 32000;  // Port each party listens to
string address =
    "127.0.0.1";      // IP address of the other party
int num_threads = 16;  // Number of threads to use
int32_t sf = 7;  // The power of scale factor (i.e. scale
                 // factor : 2 ** sf)
string use = "";
int num = 1000;

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
  amap.arg("use", use);
  amap.arg("num", num);
  amap.parse(argc, argv);

  /***** Crypto Info *****/
  BfvCiphertext ct_temp = context.new_ciphertext(1, level);
  ciphersize = ct_temp.serialize(*param).size();
  FpgaDevice::init();
  param->print();
  context_p = &context;
  CryptoInfo info;

  vector<BfvCiphertext> ac_ct;
  vector<BfvPlaintextRingt> w_pt;
  vector<BfvCiphertext> out_ct;
  for (int j = 0; j < num; j++) {
       vector<uint64_t> tmp;
       for (int i = 0; i < N; i++) {
              tmp.push_back(i);
       }
       w_pt.push_back(context_p->encode_ringt(tmp));
       ac_ct.push_back(context_p->encrypt_asymmetric(context_p->encode(tmp, level)));
       out_ct.push_back(std::move(context_p->new_ciphertext(1, level)));
  }

  if (use == "rot") {
    std::shared_ptr<FpgaProject> project = std::make_shared<FpgaProject>(string("../../tests/test_rotation"));
    vector<CxxVectorArgument> cxx_args = {{"ac", &ac_ct}, {"y", &out_ct} };
    auto computation_run_time = project->run(context_p, cxx_args, true);
    std::cout << "FPGA run time in second = " << static_cast<double>(computation_run_time) / (1000000000.0) << std::endl;
  }
  else if (use == "mult") {
    std::shared_ptr<FpgaProject> project = std::make_shared<FpgaProject>(string("../../tests/test_mult"));
    vector<CxxVectorArgument> cxx_args = {{"w", &w_pt}, {"ac", &ac_ct}, {"y", &out_ct} };
    auto computation_run_time = project->run(context_p, cxx_args, true);
    std::cout << "FPGA run time in second = " << static_cast<double>(computation_run_time) / (1000000000.0) << std::endl;
  }
}