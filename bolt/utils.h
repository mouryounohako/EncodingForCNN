#include "cxx_fhe_lib_v2.h"
#include "library_fixed.h"

#ifndef UTILS_H
#define UTILS_H
using namespace cxx_sdk_v2;

#define Arr1DIdx(s0, i) (i)
#define Arr2DIdx(s0, s1, i, j) (((i) * (s1) + (j)))
#define Arr3DIdx(s0, s1, s2, i, j, k) \
  (((i) * (s1) * (s2) + (j) * (s2) + (k)))
#define Arr4DIdx(s0, s1, s2, s3, i, j, k, l)       \
  (((i) * (s1) * (s2) * (s3) + (j) * (s2) * (s3) + \
    (k) * (s3) + (l)))
#define Arr5DIdx(s0, s1, s2, s3, s4, i, j, k, l, m) \
  (((i) * (s1) * (s2) * (s3) * (s4) +               \
    (j) * (s2) * (s3) * (s4) + (k) * (s3) * (s4) +  \
    (l) * (s4) + (m)))

/*
 * Transfer context
 *
 */
BfvContext transfer_context(int party,
                            BfvContext* context_p) {
  if(party == sci::BOB) {
    std::cout << "-----------Send Public Context-----------"
              << std::endl;
    int ctx_size = 0;
    BfvContext pub_context =
        context_p->make_public_context();
    vector<uint8_t> ctx_ser = pub_context.serialize();
    ctx_size = ctx_ser.size();
    std::cout << ctx_size << std::endl;
    io->send_data(&ctx_size, 1 * sizeof(int));
    io->send_data(static_cast<void*>(ctx_ser.data()),
                  ctx_size);
    std::cout << "-----------Public Context Sent-----------"
              << std::endl;
    return std::move(pub_context);
  } else {
    std::cout << "-----------Recv Public Context-----------"
              << std::endl;
    int ctx_size = 0;
    io->recv_data(&ctx_size, 1 * sizeof(int));
    vector<uint8_t> ctx(ctx_size, 0);
    io->recv_data(static_cast<void*>(ctx.data()), ctx_size);
    BfvContext ctx_buffer = (BfvContext::deserialize(&ctx));
    std::cout
        << "-----------Public Context Recved-----------"
        << std::endl;
    return std::move(ctx_buffer);
  }
}

class CryptoInfo {
 public:
  BfvContext* context_p;
  int party;        // Party ID
  int level;        // Polynomial level
  int cipher_size;  // Size od one ciphertext after
                    // serialize
  int prime_mod;
  int N;  // Polynomial degree
  CryptoInfo() {}
  CryptoInfo(int party, BfvContext* context_p = nullptr,
             int level = 1, int cipher_size = 0,
             int prime_mod = 1, int N = 8192) {
    this->party = party;
    this->context_p = context_p;
    this->level = level;
    this->cipher_size = cipher_size;
    this->prime_mod = prime_mod;
    this->N = N;
  }
  CryptoInfo(const CryptoInfo& other) {
    this->party = other.party;
    this->context_p = other.context_p;
    this->level = other.level;
    this->cipher_size = other.cipher_size;
    this->prime_mod = other.prime_mod;
    this->N = other.N;
  }
};

template <typename Data>
void SSreconstruct(Tensor<Data>& input, int party,
                   int prime_mod) {
  if(party == sci::ALICE) {
    io->send_data(input.cached_data.data(),
                  input.size() * sizeof(Data));
  } else {
    Tensor<uint64_t> output(input.tensorshape);
    io->recv_data(output.cached_data.data(),
                  input.size() * sizeof(Data));
    input += output;
    input %= prime_mod;

    /*
    for (int i = 0; i < input.size() / input.tensorshape.shape[1]; i++){
      for (int j = 0; j < input.tensorshape.shape[1]; j++){
        std::cout << input.cached_data[i * input.tensorshape.shape[1] + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << input.size();
    */
  }
}

template <typename Data>
Tensor<Data> Relu(Tensor<Data> input, int sf,
                  bool doTruncation) {
  int32_t size = input.size();
  Tensor<Data> output(input.tensorshape);
  Relu(size, input.cached_data.data(),
       output.cached_data.data(), sf, doTruncation);
  return output;
}

template <typename Data>
Tensor<Data> MaxPool(int32_t C, int32_t H, int32_t W,
                     int32_t kh, int32_t padding,
                     int32_t stride, Tensor<Data> input) {
  int32_t Ho = (H - kh + 2 * padding) / stride + 1;
  int32_t Wo = (W - kh + 2 * padding) / stride + 1;
  Tensor<Data> out(
      TensorShape(3, {C, Ho, Wo}));
   MaxPool(1, Ho, Wo, C, kh, kh, padding,
              padding, padding, padding, stride,
              stride, 1, H, W, C,
              input.cached_data.data(),
              out.cached_data.data());
  return out;
}
#endif