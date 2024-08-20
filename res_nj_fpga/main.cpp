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
int32_t bitlength = 12;  // Bitlength of the plaintext
// const uint64_t t = sci::default_prime_mod.at(bitlength);
const uint64_t t = 0x1b4001;
// const uint64_t t = 1 << 21;
std::shared_ptr<BfvParameter> param = std::make_shared<BfvParameter>(BfvParameter::create_fpga_parameter(t));
BfvContext *pub_context_p, *context_p;
BfvContext context =
    BfvContext::create_random_context(*param);
vector<string> fpgapath = {
    string("../../res_nj_fpga/"
           "coeff_resnet18_fpga_10"),
    string("../../res_nj_fpga/"
           "neujeans0"),
    string("../../res_nj_fpga/"
           "neujeans0"),
    string("../../res_nj_fpga/"
           "neujeans0"),
    string("../../res_nj_fpga/"
           "neujeans0"),
    string("../../res_nj_fpga/"
           "neujeans1"),
    string("../../res_nj_fpga/"
           "neujeans2"),
    string("../../res_nj_fpga/"
           "neujeans3"),
    string("../../res_nj_fpga/"
           "neujeans2"),
    string("../../res_nj_fpga/"
           "neujeans2"),
    string("../../res_nj_fpga/"
           "neujeans4"),
    string("../../res_nj_fpga/"
           "neujeans5"),
    string("../../res_nj_fpga/"
           "neujeans6"),
    string("../../res_nj_fpga/"
           "neujeans5"),
    string("../../res_nj_fpga/"
           "neujeans5"),
    string("../../res_nj_fpga/"
           "neujeans7"),
    string("../../res_nj_fpga/"
           "neujeans8"),
    string("../../res_nj_fpga/"
           "neujeans9"),
    string("../../res_nj_fpga/"
           "neujeans8"),
    string("../../res_nj_fpga/"
           "neujeans8"),
};
vector<std::shared_ptr<FpgaProject>> project_ls = {
       std::make_shared<FpgaProject>(fpgapath[0]),
       std::make_shared<FpgaProject>(fpgapath[1]),
       std::make_shared<FpgaProject>(fpgapath[2]),
       std::make_shared<FpgaProject>(fpgapath[3]),
       std::make_shared<FpgaProject>(fpgapath[4]),
       std::make_shared<FpgaProject>(fpgapath[5]),
       std::make_shared<FpgaProject>(fpgapath[6]),
       std::make_shared<FpgaProject>(fpgapath[7]),
       std::make_shared<FpgaProject>(fpgapath[8]),
       std::make_shared<FpgaProject>(fpgapath[9]),
       std::make_shared<FpgaProject>(fpgapath[10]),
       std::make_shared<FpgaProject>(fpgapath[11]),
       std::make_shared<FpgaProject>(fpgapath[12]),
       std::make_shared<FpgaProject>(fpgapath[13]),
       std::make_shared<FpgaProject>(fpgapath[14]),
       std::make_shared<FpgaProject>(fpgapath[15]),
       std::make_shared<FpgaProject>(fpgapath[16]),
       std::make_shared<FpgaProject>(fpgapath[17]),
       std::make_shared<FpgaProject>(fpgapath[18]),
       std::make_shared<FpgaProject>(fpgapath[19]),
};
/* Global variable for comm. */
int party = 0;     // 1 for Alice(Server), 2 for Bob(Client)
int port = 32000;  // Port each party listens to
string address =
    "127.0.0.1";      // IP address of the other party
int num_threads = 16;  // Number of threads to use
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
  ciphersize = ct_temp.serialize(*param).size();
  if(party == sci::ALICE) {
    FpgaDevice::init();
  }
  /***** Setup Phase *****/
  StartComputation();
  /***** context transfer *****/
  param->print();
  context_p = &context;
  auto pub_context = transfer_context(party, context_p);
  pub_context_p = &pub_context;
  /***** Computation Phase *****/
  CryptoInfo info;

  if(party == sci::ALICE) {
    info = CryptoInfo(party,param, pub_context_p, level,
                      ciphersize, prime_mod, N);
    for(int i = 0; i < num_threads; i++) {
      auto temp = std::make_shared<BfvContext>(
          pub_context_p->shallow_copy_context());
      info.context_par.push_back(temp);
    }
  } else {
    info = CryptoInfo(party,param, context_p, level, ciphersize,
                      prime_mod, N);
    for(int i = 0; i < num_threads; i++) {
      auto temp = std::make_shared<BfvContext>(
          context_p->shallow_copy_context());
      info.context_par.push_back(temp);
    }
  }

  ResNet18CoeffFPGA<uint64_t> resnet18_coeff(
      vector<string>(20, ""), info, project_ls);
  Tensor<uint64_t> input(TensorShape(1, {3, 224, 224}));
  Tensor<uint64_t> output =
      resnet18_coeff.forward(input, sf);
  SSreconstruct<uint64_t>(output, party, prime_mod);

  EndComputation();
}