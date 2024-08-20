#include <inttypes.h>

#include <random>
#include <cmath>

#include "cxx_fhe_lib_v2.h"
#include "cxx_fpga_ops_v2.h"
#include "library_fixed.h"
#include "tensor.h"
#include "utils.h"

#ifndef MODULE_H
#define MODULE_H
/*
 * Module object, the top level of all operators.
 * @param Data: the data type of the underlying data, like
 * uint64_t.
 */

using namespace cxx_sdk_v2;
using namespace std;

/*
 * Base class for all module operator.
 * @param Data: the data type of the underlying data, like
 * uint64_t.
 */
template <typename Data>
class Module {
 public:
  Module(){};
  virtual ~Module(){};

  virtual void pack_ac(){};
  virtual void depack_res(){};
  virtual Tensor<Data> forward() { return Tensor<Data>(); };
};

/*
 * Conv2dCoeff operator, using the coefficient packing
 * method.
 * @param Data: the data type of the underlying data, like
 * uint64_t.
 */
template <typename Data>
class Conv2dCoeff : public Module<Data> {
 public:
  /* Tensor info */
  Tensor<Data> weight;
  int Co, Ci, H, W, kh, stride, padding, Ho, Wo;
  /* Crypto info */
  int channel_per_poly;
  int num_poly;
  int tile;
  std::vector<BfvPlaintextMul> weight_pt;
  CryptoInfo ct_info;
  BfvParameter param;

  Conv2dCoeff(){};
  /*
   * Constructor of Conv2d.
   * Server will load the weight tensor and pack the weight
   * into polynomials.
   */

  Conv2dCoeff(Tensor<Data>& weight, int Co, int Ci, int H,
              int W, int kh, int stride, int padding,
              CryptoInfo crypto_info, BfvParameter param)
      : Module<Data>(),
        weight(weight),
        Co(Co),
        Ci(Ci),
        H(H),
        W(W),
        kh(kh),
        stride(stride),
        padding(padding),
        ct_info(crypto_info), 
        param(std::move(param)) {
    Ho = (H - kh + 2 * padding) / stride + 1,
    Wo = (W - kh + 2 * padding) / stride + 1;

    channel_per_poly = 0;
    int paddedH = H + 2 * padding,
        paddedW = W + 2 * padding;
    tile = 1;
    int Oc = 0, kh1 = kh / 2, Sw = W + 2 * padding,
        Sh = H + 2 * padding;

    for(int c = 0; c < Ci; c++) {
      Oc = c * (Sh + 2 * kh1) * (Sw + 2 * kh1) +
           (kh - 1) * (Sw + 2 * kh1) + kh - 1 +
           (Sw + 2 * kh1) * (Sh - 1) + Sw - 1;
      if(Oc < ct_info.N) {
        channel_per_poly++;
      } else {
        break;
      }
    }

    if(channel_per_poly == 0) {
      channel_per_poly = 1;
      while(true) {
        tile++;
        Sw = ((W + 2 * padding) + tile - 1) / tile;
        Sh = ((H + 2 * padding) + tile - 1) / tile;
        Oc = (kh - 1) * (Sw + 2 * kh1) + kh - 1 +
             (Sw + 2 * kh1) * (Sh - 1) + Sw - 1;
        if(Oc < ct_info.N) {
          break;
        }
      }
    }

    num_poly =
        (Ci + channel_per_poly - 1) / channel_per_poly;

    if(party == sci::ALICE) {
      // load the weight tensor

      for(int co = 0; co < Co; co++) {
        for(int p = 0; p < num_poly; p++) {
          int channel_start = p * channel_per_poly;
          std::vector<uint64_t> w_msg(ct_info.N, 0);

          int O = (channel_per_poly - 1) * (Sh + 2 * kh1) *
                      (Sw + 2 * kh1) +
                  (kh - 1) * (Sw + 2 * kh1) + kh - 1;

          for(int ci = 0; ci < channel_per_poly; ci++) {
            for(int ki = 0; ki < kh; ki++) {
              for(int kj = 0; kj < kh; kj++) {
                if(ci + channel_start < Ci &&
                   ci + channel_start >= 0) {
                  w_msg[O -
                        ci * (Sh + 2 * kh1) *
                            (Sw + 2 * kh1) -
                        ki * (Sw + 2 * kh1) - kj] =
                      weight.cached_data[co * Ci * kh * kh +
                                         (ci +
                                          channel_start) *
                                             kh * kh +
                                         ki * kh + kj];
                }
              }
            }
          }
          weight_pt.push_back(
              ct_info.context_p->encode_coeffs_mul(
                  w_msg, ct_info.level));
        }
      }
    }
  };
  /*
   * Pack the activation tensor into polynomials.
   */
  vector<vector<Data>> pack_ac(Tensor<Data>& unpadded_ac) {
    Tensor<Data> activation(TensorShape(
        3,
        {unpadded_ac.tensorshape.shape[0],
         unpadded_ac.tensorshape.shape[1] + 2 * padding,
         unpadded_ac.tensorshape.shape[2] + 2 * padding}));

    int paddedH = H + 2 * padding;
    int paddedW = W + 2 * padding;

    for(int c = 0; c < activation.tensorshape.shape[0];
        ++c) {
      for(int h = 0; h < H; ++h) {
        for(int w = 0; w < W; ++w) {
          // Calculate the index in the original (flattened)
          // 3D array
          int ac_index = c * H * W + h * W + w;
          // Calculate the index in the padded (flattened)
          // 3D array Note: h and w are offset by p to
          // account for padding
          int padded_index = c * paddedH * paddedW +
                             (h + padding) * paddedW +
                             (w + padding);
          // Copy the value from the original array to the
          // padded array
          activation.cached_data[padded_index] =
              unpadded_ac.cached_data[ac_index];
        }
      }
    }

    std::vector<vector<Data>> packed_ac;
    int Sh = ((H + 2 * padding) + tile - 1) / tile;
    int Sw = ((W + 2 * padding) + tile - 1) / tile;
    int kh1 = kh / 2;
    for(int ti = 0; ti < tile; ti++) {
      for(int tj = 0; tj < tile; tj++) {
        for(int p = 0; p < num_poly; p++) {
          int channel_start = p * channel_per_poly;
          std::vector<uint64_t> mask_msg(ct_info.N, 0);
          for(int c = 0; c < channel_per_poly; c++) {
            for(int i = 0; i < Sh + 2 * kh1; i++) {
              for(int j = 0; j < Sw + 2 * kh1; j++) {
                if(c + channel_start < Ci &&
                   i + ti * Sh - kh1 < H + 2 * padding &&
                   j + tj * Sw - kh1 < W + 2 * padding &&
                   c + channel_start >= 0 &&
                   i + ti * Sh - kh1 >= 0 &&
                   j + tj * Sw - kh1 >= 0) {
                  mask_msg[(Sh + 2 * kh1) * (Sw + 2 * kh1) *
                               c +
                           (Sw + 2 * kh1) * i + j] =
                      activation.cached_data
                          [(W + 2 * padding) *
                               (H + 2 * padding) *
                               (c + channel_start) +
                           (W + 2 * padding) *
                               (i + ti * Sh - kh1) +
                           (j + tj * Sw - kh1)];
                } else {
                  mask_msg[(Sh + 2 * kh1) * (Sw + 2 * kh1) *
                               c +
                           (Sw + 2 * kh1) * i + j] = 0;
                }
              }
            }
          }
          packed_ac.push_back(mask_msg);
        }
      }
    }
    return packed_ac;
  };

  /*
   * depack the result ciphertext into tensor.
   */

  Tensor<Data> depack_res(vector<vector<Data>>& res_poly) {
    int Sh = ((H + 2 * padding) + tile - 1) / tile;
    int Sw = ((W + 2 * padding) + tile - 1) / tile;
    int O = (channel_per_poly - 1) * (Sh + 2 * padding) *
                (Sw + 2 * padding) +
            (kh - 1) * (Sw + 2 * padding) + kh - 1;

    Tensor<Data> out(TensorShape(3, {Co, Ho, Wo}));

    for(int c = 0; c < Co; c++) {
      for(int i = 0; i < Ho; i++) {
        for(int j = 0; j < Wo; j++) {
          int ti = (i * stride + kh / 2) / Sh;
          int tj = (j * stride + kh / 2) / Sw;
          int ii = (i * stride + kh / 2) % Sh;
          int jj = (j * stride + kh / 2) % Sw;

          out.cached_data[c * Ho * Wo + i * Wo + j] =
              res_poly[c * tile * tile + ti * tile + tj]
                      [O + (Sw + 2 * padding) * ii + jj];
        }
      }
    }
    return out;
  }

  /*
   * Forward the convolution operator.
   */
  Tensor<Data> forward(Tensor<Data>& input) {

#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif

    // Packing then combine the mask
    vector<BfvCiphertext> ac_ct;  // For client
    vector<BfvPlaintext> ac_pt;   // For server
    vector<vector<Data>> ac_msg;
    vector<BfvCiphertext> out_ct_ls;
    vector<vector<Data>> out_msg_ls;
    vector<BfvPlaintext> out_mask;

    if(party == sci::ALICE) {
      // pack the activation tensor
      ac_msg = pack_ac(input);
      for(int i = 0; i < ac_msg.size(); i++) {
        ac_pt.push_back(ct_info.context_p->encode_coeffs(
            ac_msg[i], ct_info.level));
      }
      for(int i = 0; i < ac_msg.size(); i++) {
        vector<uint8_t> buffer(ct_info.cipher_size, 0);
        io->recv_data(static_cast<void*>(buffer.data()),
                      ct_info.cipher_size);
        BfvCiphertext ct_buffer =
            BfvCiphertext::deserialize(&buffer);
        ac_ct.push_back(move(ct_buffer));
      }
      for(int i = 0; i < ac_msg.size(); i++) {
        ct_info.context_p->add_plain_inplace(ac_ct[i],
                                             ac_pt[i]);
      }
    } else {
      // for client
      ac_msg = pack_ac(input);
      for(int i = 0; i < ac_msg.size(); i++) {
        ac_ct.push_back(
            ct_info.context_p->encrypt_asymmetric(
                ct_info.context_p->encode_coeffs(
                    ac_msg[i], ct_info.level)));
      }
      for(int i = 0; i < ac_msg.size(); i++) {
        io->send_data(
            static_cast<void*>(ac_ct[i].serialize(param).data()),
            ct_info.cipher_size);
      }
    }
#ifdef LOG_LAYERWISE
    auto temp1 = TIMER_TILL_NOW;
#endif
    // Perform computation
    if(party == sci::ALICE) {
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            vector<uint64_t> all_zeros(8192, 0);
            BfvPlaintext res_pt =
                ct_info.context_p->encode_coeffs(
                    all_zeros, ct_info.level);
            BfvCiphertext res_ct =
                ct_info.context_p->encrypt_asymmetric(
                    res_pt);
            for(int p = 0; p < num_poly; p++) {
              ct_info.context_p->add_inplace(
                  res_ct,
                  ct_info.context_p->mult_plain_mul(
                      ac_ct[Arr3DIdx(tile, tile, num_poly,
                                     ti, tj, p)],
                      weight_pt[Arr2DIdx(Co, num_poly, co,
                                         p)]));
            }
            out_ct_ls.push_back(move(res_ct));
          }
        }
      }
    }
#ifdef LOG_LAYERWISE
    auto temp2 = TIMER_TILL_NOW;
    Conv2dCompTimeInMilliSec += (temp2 - temp1);
#endif

    /***** Generate the secret share *****/
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> distrib(
        0, ct_info.prime_mod - 1);

    if(party == sci::ALICE) {
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            // TODO: needs more secure PRG
            vector<uint64_t> pos_mask(ct_info.N, 0);
            vector<uint64_t> neg_mask(ct_info.N, 0);
            for(int i = 0; i < pos_mask.size(); i++) {
              pos_mask[i] = distrib(gen);
              neg_mask[i] = ct_info.prime_mod - pos_mask[i];
            }
            BfvPlaintext temp_buffer_pos =
                ct_info.context_p->encode_coeffs(
                    pos_mask, ct_info.level);
            BfvPlaintext temp_buffer_neg =
                ct_info.context_p->encode_coeffs(
                    neg_mask, ct_info.level);
            ct_info.context_p->add_plain_inplace(
                out_ct_ls[Arr3DIdx(Co, tile, tile, co, ti,
                                   tj)],
                temp_buffer_neg);
            out_mask.push_back(move(temp_buffer_pos));
          }
        }
      }
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            io->send_data(
                static_cast<void*>(
                    out_ct_ls[(Arr3DIdx(Co, tile, tile, co,
                                        ti, tj))]
                        .serialize(param)
                        .data()),
                ct_info.cipher_size);
          }
        }
      }
    } else {
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            vector<uint8_t> buffer(ct_info.cipher_size, 0);
            io->recv_data(static_cast<void*>(buffer.data()),
                          ct_info.cipher_size);
            BfvCiphertext ct_buffer =
                BfvCiphertext::deserialize(&buffer);
            out_ct_ls.push_back(move(ct_buffer));
          }
        }
      }
    }

    /***** decrypt and unpack *****/
    if(party == sci::ALICE) {
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            vector<uint64_t> res =
                ct_info.context_p->decode_coeffs(
                    out_mask[Arr3DIdx(Co, tile, tile, co,
                                      ti, tj)]);
            out_msg_ls.push_back(res);
          }
        }
      }
    } else {
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            vector<uint64_t> res =
                ct_info.context_p->decode_coeffs(
                    ct_info.context_p->decrypt(
                        out_ct_ls[Arr3DIdx(Co, tile, tile,
                                           co, ti, tj)]));
            out_msg_ls.push_back(res);
          }
        }
      }
    }

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    Conv2dCoeffTimeInMilliSec += temp;
    std::cout << "Time in sec for current relu = "
              << (temp / 1000.0) << std::endl;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    Conv2dCoeffCommSent += curComm;
#endif

    return depack_res(out_msg_ls);
  }
};

/*
 * activation-weight multiplication, using BOLT packing with BSGS
 * there should be no limitation for the size of matrices
*/

template <typename Data>
class BoltMatmul : public Module<Data>{
  public:
  Tensor<Data> weight;
  int d0_unp, d1_unp, d2_unp, d0, d1, d2, t, input_rot;
  int num_poly;
  CryptoInfo ct_info;
  std::vector<BfvPlaintextMul> weight_pt;
  BfvParameter param;
  BoltMatmul(){};

  BoltMatmul(Tensor<Data>& weight, int d0_unp, int d1_unp, int d2_unp, CryptoInfo crypto_info, BfvParameter param)
      : Module<Data>(),
        weight(weight),
        d0_unp(d0_unp),
        d1_unp(d1_unp),
        d2_unp(d2_unp),
        ct_info(crypto_info), 
        param(std::move(param)){
    if (crypto_info.N % d0_unp == 0){
      d0 = d0_unp;
    }
    else{
      // round up the d0 to the nearest power of 2
      double lnd0 = log(d0_unp) / log(2);
      d0 = std::pow(2, static_cast<int>(lnd0) + 1);
    }
    d1 = d1_unp;
    d2 = d2_unp;
    t = crypto_info.N / d0; // number of rows in the polynomial
    
    if (party == sci::ALICE){
      assert(weight.tensorshape.shape[0] == d1_unp && weight.tensorshape.shape[1] == d2_unp);
      if (d1_unp % t != 0){
        d1 = d1_unp + t - d1_unp % t;  // round up to the nearest multiple of t
        std::vector<Data> tmp1(d1 * d2_unp, 0);
        for (int i = 0; i < weight.cached_data.size(); i++){
          tmp1[i] = weight.cached_data[i];
        }
        weight.cached_data = tmp1;
        weight.tensorshape.shape[0] = d1;
      }
      if (d2_unp % t != 0){
        // round up d2 to the nearest multiple of 2t, this is due to the rotation property of SIMD encoding
        // 8192 slots need to be treated as 2*4096 slots
        d2 = d2_unp + t - d2_unp % t;
        std::vector<Data> tmp2(d1 * d2, 0);
        for (int i = 0; i < weight.cached_data.size(); i++){
          tmp2[(i / d2_unp) * d2 + i % d2_unp] = weight.cached_data[i];
        }
        weight.cached_data = tmp2;
        weight.tensorshape.shape[1] = d2;
      }
    }
    else {
      if (d1_unp % t != 0){
        d1 = d1_unp + t - d1_unp % t;
      }
      if (d2_unp % t != 0){
        d2 = d2_unp + t - d2_unp % t;
      }
    }

    // Compute input rotation;
    // Formula for total rotations are d1/t * input_rot + d2/2t * output_rot
    // (input_rot+1) (output_rot+1) = t
    double p =
        std::log(std::sqrt(d2 * t / d1)) / std::log(2);
    input_rot = std::pow(2, static_cast<int>(p));
    if (d1 * input_rot + d2 * t / input_rot > 2 * d1 * input_rot + d2 * t / (2 * input_rot)){
      input_rot *= 2; // Why must power of two?
    }
    
    if (party == sci::ALICE){
      for (int i = 0; i < d1 / t; i++){
        for (int j = 0; j < d2 / t; j++){
          std::vector<uint64_t> w_msg(ct_info.N, 0);
          for (int k = 0; k < t; k++){
            for (int l = 0; l < ct_info.N / 2; l++){
              //TODO: the exact meaning of the following line
              w_msg[l] = weight.cached_data[(i * t + (l / d0 + input_rot - 1 - (k % input_rot)) % t) * d2 + (j * t + (3 * t - 1 - l / d0 -input_rot + 1 + (k % input_rot) - k) % t)]; // i * d2 + j * t + k
              w_msg[l + ct_info.N / 2] = weight.cached_data[(i * t + (l / d0 + input_rot - 1 - (k % input_rot)) % t) * d2 + (j * t + (3 * t - 1 - l / d0 -input_rot + 1 + (k % input_rot) - k) % t)];
            }
            weight_pt.push_back(ct_info.context_p->encode_mul(w_msg, ct_info.level));
          }
        }
      }
    }
  }

  vector<vector<Data>> pack_ac(Tensor<Data>& act){
    assert(act.tensorshape.shape[0] == d0_unp && act.tensorshape.shape[1] == d1_unp);
    // Round up d0 to the nearest PO2
    // Round up d1 to the nearest multiple of t; t is multiple of PO2
    if(d0 != d0_unp) {
      std::vector<Data> tmp0(d0 * d1_unp, 0);
      for (int i = 0; i < act.cached_data.size(); i++){
        tmp0[i] = act.cached_data[i];
      }
      act.cached_data = tmp0;
      act.tensorshape.shape[0] = d0;
    }
    if (d1_unp % t != 0){
      std::vector<Data> tmp1(d0 * d1, 0);
      for (int i = 0; i < act.cached_data.size(); i++){
        tmp1[(i / d1_unp) * d1 + i % d1_unp] = act.cached_data[i];
      }
      act.cached_data = tmp1;
      act.tensorshape.shape[1] = d1;
    }

    std::vector<vector<Data>> packed_ac;
    for (int i = 0; i < d1 / t; i++){
      // Is there a way for more efficient use of activation 
      std::vector<uint64_t> ac_msg(ct_info.N, 0);
      for (int j = 0; j < ct_info.N / 2; j++){
        ac_msg[j] = act.cached_data[(j % (d0 / 2)) * d1 + i * t + j / (d0 / 2)];
        ac_msg[j + ct_info.N / 2] = act.cached_data[(j % (d0 / 2) + d0 / 2) * d1 + i * t + j / (d0 / 2)];
      }
      packed_ac.push_back(ac_msg);
    }
    return packed_ac;
  }

  Tensor<Data> depack_res(vector<vector<Data>>& out_msg){
    Tensor<Data> out(TensorShape(2, {d0, d2}));
    for (int i = 0; i < d2 / t; i++){
      for (int j = 0; j < d0 * t; j++){
        if (j < d0 * t / 2){
          //TODO: the exact meaning of the following line
          out.cached_data[(j % (d0 / 2)) * d2 + i * t + (t - j / (d0 / 2)) % t] = out_msg[i][j];
        }
        else{
          out.cached_data[(j % (d0 / 2) + d0 / 2) * d2 + i * t + (2 * t - j / (d0 / 2)) % t] = out_msg[i][j];
        }
      }
    }

    if (d2 != d2_unp){
      std::vector<Data> tmp2(d0 * d2_unp, 0);
      for (int i = 0; i < d0; i++){
        for (int j = 0; j < d2_unp; j++){
          tmp2[i * d2_unp + j] = out.cached_data[i * d2 + j];
        }
      }
      out.cached_data = tmp2;
      out.tensorshape.shape[1] = d2_unp;
    }
    if (d0 != d0_unp){
      std::vector<Data> tmp0(d0_unp * d2_unp, 0);
      for (int i = 0; i < d0_unp * d2_unp; i++){
        tmp0[i] = out.cached_data[i];
      }
      out.cached_data = tmp0;
      out.tensorshape.shape[0] = d0_unp;
    }
    return out;
  }

  Tensor<Data> forward(Tensor<Data>& input){

#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
    vector<BfvCiphertext> ac_ct;
    vector<BfvPlaintext> ac_pt;
    vector<vector<Data>> ac_msg;
    vector<BfvCiphertext> int_ct; // intermediate product in ciphertext
    vector<BfvCiphertext> out_ct;
    vector<BfvPlaintext> out_share;
    vector<vector<Data>> out_msg;
    
    if (party == sci::ALICE){
      // Combine the mask
      ac_msg = pack_ac(input);
      for (int i = 0; i < ac_msg.size(); i++) {
        ac_pt.push_back(ct_info.context_p->encode(ac_msg[i], ct_info.level));
      }
      for (int i = 0; i < ac_msg.size(); i++) {
        vector<uint8_t> buffer(ct_info.cipher_size, 0);
        io->recv_data(static_cast<void*>(buffer.data()), ct_info.cipher_size);
        BfvCiphertext ct_buffer = BfvCiphertext::deserialize(&buffer);
        ac_ct.push_back(move(ct_buffer));
      }
      for (int i = 0; i < ac_msg.size(); i++) {
        ct_info.context_p->add_plain_inplace(ac_ct[i], ac_pt[i]);
      }
    } 
    else{
      ac_msg = pack_ac(input);
      for (int i = 0; i < ac_msg.size(); i++) {
        ac_ct.push_back(ct_info.context_p->encrypt_asymmetric(ct_info.context_p->encode(ac_msg[i], ct_info.level)));
      }
      for (int i = 0; i < ac_msg.size(); i++) {
        io->send_data(static_cast<void*>(ac_ct[i].serialize(param).data()), ct_info.cipher_size);
      }
    }
#ifdef LOG_LAYERWISE
    auto temp1 = TIMER_TILL_NOW;
#endif
    if (party == sci::ALICE){
      for (int i = 0; i < d2 / t; i++){
        out_ct.push_back(std::move(ct_info.context_p->new_ciphertext(1, ct_info.level)));
      }
      // std::cout << "parameters: " << d0 << " " << d1 << " " << d2 << " " << t < " " << input_rot << std::endl;
      FpgaDevice::init();
      FpgaProject fpga_project("../../bolt_fpga/bolt");
      vector<CxxVectorArgument> cxx_args = {{"w", &weight_pt}, {"ac", &ac_ct}, {"y", &out_ct} };
      fpga_project.run(ct_info.context_p, cxx_args, true);
    }
#ifdef LOG_LAYERWISE
    auto temp2 = TIMER_TILL_NOW;
    Conv2dCompTimeInMilliSec += (temp2 - temp1); // kept the variable name
#endif
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> distrib(0, ct_info.prime_mod - 1);
    if (party == sci::ALICE){
      for (int i = 0; i < d2 / t; i++){
        vector<uint64_t> pos_mask(ct_info.N, 0);
        vector<uint64_t> neg_mask(ct_info.N, 0);
        for(int j = 0; j < pos_mask.size(); j++) {
          pos_mask[j] = distrib(gen);
          neg_mask[j] = ct_info.prime_mod - pos_mask[j];
        }
        BfvPlaintext temp_buffer_pos = ct_info.context_p->encode(pos_mask, ct_info.level);
        BfvPlaintext temp_buffer_neg = ct_info.context_p->encode(neg_mask, ct_info.level);
        ct_info.context_p->add_plain_inplace(out_ct[i], temp_buffer_neg);
        out_share.push_back(move(temp_buffer_pos));
        io->send_data(static_cast<void*>(out_ct[i].serialize(param).data()), ct_info.cipher_size);
      }
    }
    else {
      for(int i = 0; i < d2 / t; i++) {
        vector<uint8_t> buffer(ct_info.cipher_size, 0);
        io->recv_data(static_cast<void*>(buffer.data()), ct_info.cipher_size);
        BfvCiphertext ct_buffer = BfvCiphertext::deserialize(&buffer);
        out_ct.push_back(move(ct_buffer));
      }
    }

    if(party == sci::ALICE) {
      for(int i = 0; i < d2 / t; i++) {
        vector<uint64_t> res = ct_info.context_p->decode(out_share[i]);
        out_msg.push_back(res);
      }
    } else {
      for(int i = 0; i < d2 / t; i++) {
        vector<uint64_t> res = ct_info.context_p->decode(ct_info.context_p->decrypt(out_ct[i]));
        out_msg.push_back(res);
      }
    }
#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    Conv2dCoeffTimeInMilliSec += temp;
    std::cout << "Time in sec for current matmul = " << (temp / 1000.0) << std::endl;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    Conv2dCoeffCommSent += curComm;
#endif
    return depack_res(out_msg);
  }
};

/*
 * ResNet18Network, using the coefficient packing
 * method.
 * @param Data: the data type of the underlying data, like
 * uint64_t.
 */
template <typename Data>
class ResNet18Coeff : public Module<Data> {
 public:
  CryptoInfo ct_info;
  Conv2dCoeff<Data> conv0, conv1, conv2, conv3, conv4,
      conv5, conv6, conv7, conv8, conv9, conv10, conv11,
      conv12, conv13, conv14, conv15, conv16, conv17,
      conv18, conv19;

  ResNet18Coeff(std::vector<std::string> weight_pth,
                CryptoInfo ct_info)
      : ct_info(ct_info) {
    // Initial block
    Tensor<Data> weight0(TensorShape(4, {64, 3, 7, 7}));
    weight0.load(weight_pth[0]);
    conv0 = Conv2dCoeff<Data>(weight0, 64, 3, 224, 224, 7,
                              2, 3, ct_info);
    // First Block
    Tensor<Data> weight1(TensorShape(4, {64, 64, 3, 3}));
    weight1.load(weight_pth[1]);
    conv1 = Conv2dCoeff<Data>(weight1, 64, 64, 56, 56, 3, 1,
                              1, ct_info);

    Tensor<Data> weight2(TensorShape(4, {64, 64, 3, 3}));
    weight2.load(weight_pth[2]);
    conv2 = Conv2dCoeff<Data>(weight2, 64, 64, 56, 56, 3, 1,
                              1, ct_info);
    Tensor<Data> weight3(TensorShape(4, {64, 64, 3, 3}));
    weight3.load(weight_pth[3]);
    conv3 = Conv2dCoeff<Data>(weight3, 64, 64, 56, 56, 3, 1,
                              1, ct_info);
    Tensor<Data> weight4(TensorShape(4, {64, 64, 3, 3}));
    weight4.load(weight_pth[4]);
    conv4 = Conv2dCoeff<Data>(weight4, 64, 64, 56, 56, 3, 1,
                              1, ct_info);

    // Second Block
    // Block 1
    Tensor<Data> weight5(TensorShape(4, {128, 64, 3, 3}));
    weight5.load(weight_pth[5]);
    conv5 = Conv2dCoeff<Data>(weight5, 128, 64, 56, 56, 3,
                              2, 1, ct_info);
    Tensor<Data> weight6(TensorShape(4, {128, 128, 3, 3}));
    weight6.load(weight_pth[6]);
    conv6 = Conv2dCoeff<Data>(weight6, 128, 128, 28, 28, 3,
                              1, 1, ct_info);
    Tensor<Data> weight7(
        TensorShape(4, {128, 64, 1, 1}));  // short cut
    weight7.load(weight_pth[7]);
    conv7 = Conv2dCoeff<Data>(weight7, 128, 64, 56, 56, 1,
                              2, 0, ct_info);

    // Block 2
    Tensor<Data> weight8(TensorShape(4, {128, 128, 3, 3}));
    weight8.load(weight_pth[8]);
    conv8 = Conv2dCoeff<Data>(weight8, 128, 128, 28, 28, 3,
                              1, 1, ct_info);
    Tensor<Data> weight9(TensorShape(4, {128, 128, 3, 3}));
    weight9.load(weight_pth[9]);
    conv9 = Conv2dCoeff<Data>(weight9, 128, 128, 28, 28, 3,
                              1, 1, ct_info);


    // Third Block
    // Block 1
    Tensor<Data> weight10(TensorShape(4, {256, 128, 3, 3}));
    weight10.load(weight_pth[10]);
    conv10 = Conv2dCoeff<Data>(weight10, 256, 128, 28, 28,
                               3, 2, 1, ct_info);
    Tensor<Data> weight11(TensorShape(4, {256, 256, 3, 3}));
    weight11.load(weight_pth[11]);
    conv11 = Conv2dCoeff<Data>(weight11, 256, 256, 14, 14,
                               3, 1, 1, ct_info);
    Tensor<Data> weight12(
        TensorShape(4, {256, 128, 3, 3}));  // short cut
    weight12.load(weight_pth[12]);
    conv12 = Conv2dCoeff<Data>(weight12, 256, 128, 28, 28,
                               1, 2, 0, ct_info);
    // Block 2
    Tensor<Data> weight13(TensorShape(4, {256, 256, 3, 3}));
    weight13.load(weight_pth[13]);
    conv13 = Conv2dCoeff<Data>(weight13, 256, 256, 14, 14,
                               3, 1, 1, ct_info);
    Tensor<Data> weight14(TensorShape(4, {256, 256, 3, 3}));
    weight14.load(weight_pth[14]);
    conv14 = Conv2dCoeff<Data>(weight14, 256, 256, 14, 14,
                               3, 1, 1, ct_info);

    // Fourth Block
    // Block 1
    Tensor<Data> weight15(TensorShape(4, {512, 256, 3, 3}));
    weight15.load(weight_pth[15]);
    conv15 = Conv2dCoeff<Data>(weight15, 512, 256, 14, 14,
                               3, 2, 1, ct_info);
    Tensor<Data> weight16(TensorShape(4, {512, 512, 3, 3}));
    weight16.load(weight_pth[16]);
    conv16 = Conv2dCoeff<Data>(weight16, 512, 512, 7, 7, 3,
                               1, 1, ct_info);
    Tensor<Data> weight17(
        TensorShape(4, {512, 256, 3, 3}));  // short cut
    weight17.load(weight_pth[17]);
    conv17 = Conv2dCoeff<Data>(weight17, 512, 256, 14, 14,
                               1, 2, 0, ct_info);

    // Block 2
    Tensor<Data> weight18(TensorShape(4, {512, 512, 3, 3}));
    weight18.load(weight_pth[18]);
    conv18 = Conv2dCoeff<Data>(weight18, 512, 512, 7, 7, 3,
                               1, 1, ct_info);
    Tensor<Data> weight19(TensorShape(4, {512, 512, 3, 3}));
    weight19.load(weight_pth[19]);
    conv19 = Conv2dCoeff<Data>(weight19, 512, 512, 7, 7, 3,
                               1, 1, ct_info);
  }
  Tensor<Data> forward(Tensor<Data> input, int sf) {
    // Block 1
    Tensor<Data> ac1 = conv0.forward(input);
    Tensor<Data> ac2 = Relu<Data>(ac1, sf, 1);
    Tensor<Data> ac3 = MaxPool(64, 112, 112, 3, 1, 2, ac2);

    // Block 2-1
    Tensor<Data> ac4 = conv1.forward(ac3);
    Tensor<Data> ac5 = Relu<Data>(ac4, sf, 1);
    Tensor<Data> ac6 = conv2.forward(ac5);
    Tensor<Data> ac7 = Relu<Data>(ac6 + ac3, sf, 1);

    // Block 2-2
    Tensor<Data> ac8 = conv3.forward(ac7);
    Tensor<Data> ac9 = Relu<Data>(ac8, sf, 1);
    Tensor<Data> ac10 = conv4.forward(ac9);
    Tensor<Data> ac11 = Relu<Data>(ac10 + ac7, sf, 1);

    // Block 3-1
    Tensor<Data> ac12 = conv5.forward(ac11);
    Tensor<Data> ac13 = Relu<Data>(ac12, sf, 1);
    Tensor<Data> ac14 = conv6.forward(ac13);
    // short cut
    Tensor<Data> ac15 =
        Relu<Data>(ac14 + conv7.forward(ac11), sf, 1);

    // Block 3-2
    Tensor<Data> ac16 = conv8.forward(ac15);
    Tensor<Data> ac17 = Relu<Data>(ac16, sf, 1);
    Tensor<Data> ac18 = conv9.forward(ac17);
    Tensor<Data> ac19 = Relu<Data>(ac18 + ac15, sf, 1);

    // Block 4-1
    Tensor<Data> ac20 = conv10.forward(ac19);
    Tensor<Data> ac21 = Relu<Data>(ac20, sf, 1);
    Tensor<Data> ac22 = conv11.forward(ac21);
    // short cut
    Tensor<Data> ac23 =
        Relu<Data>(ac22 + conv12.forward(ac19), sf, 1);

    // Block 4-2
    Tensor<Data> ac24 = conv13.forward(ac23);
    Tensor<Data> ac25 = Relu<Data>(ac24, sf, 1);
    Tensor<Data> ac26 = conv14.forward(ac25);
    Tensor<Data> ac27 = Relu<Data>(ac26 + ac23, sf, 1);

    // Block 5-1
    Tensor<Data> ac28 = conv15.forward(ac27);
    Tensor<Data> ac29 = Relu<Data>(ac28, sf, 1);
    Tensor<Data> ac30 = conv16.forward(ac29);
    // short cut
    Tensor<Data> ac31 =
        Relu<Data>(ac30 + conv17.forward(ac27), sf, 1);

    // Block 5-2
    Tensor<Data> ac32 = conv18.forward(ac31);
    Tensor<Data> ac33 = Relu<Data>(ac32, sf, 1);
    Tensor<Data> ac34 = conv19.forward(ac33);
    Tensor<Data> ac35 = Relu<Data>(ac34 + ac31, sf, 1);

    return ac35;
  }
};
#endif  // MODULE_H