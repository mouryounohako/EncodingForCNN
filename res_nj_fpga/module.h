#include <inttypes.h>

#include <random>
#include <memory>

#include "hexl/ntt/ntt.hpp"
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
 * Conv2dCoeffFPGA operator, using the coefficient packing
 * method with FPGA acceleration.
 * @param Data: the data type of the underlying data, like
 * uint64_t.
 */
template <typename Data>
class Conv2dCoeffFPGA : public Module<Data> {
 public:
  /* Tensor info */
  Tensor<Data> weight;
  int Co, Ci, H, W, kh, stride, padding, Ho, Wo;
  /* Crypto info */
  int channel_per_poly;
  int num_poly;
  int tile;
  std::vector<BfvPlaintextRingt> weight_pt;
  CryptoInfo ct_info;
  // std::string pth;
  std::shared_ptr<FpgaProject> project;

  Conv2dCoeffFPGA(){};
  /*
   * Constructor of Conv2d.
   * Server will load the weight tensor and pack the weight
   * into polynomials.
   */

  Conv2dCoeffFPGA(Tensor<Data>& weight, int Co, int Ci,
                  int H, int W, int kh, int stride,
                  int padding, CryptoInfo crypto_info,
                  // std::string pth
                  std::shared_ptr<FpgaProject>& project
                  )
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
        // pth(pth) 
        project(project)
        {
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

    // std::cout<< "#input poly num: " <<num_poly * tile * tile<<std::endl;
    // std::cout<< "#output poly num: " <<Co * tile * tile<<std::endl;
    // std::cout<< "#mult poly num: " <<Co * tile * tile * num_poly<<std::endl;

    if(party == sci::ALICE) {
      // load the weight tensor

      for(int co = 0; co < Co; co++) {
        for(int p = 0; p < num_poly; p++) {
          int channel_start = p * channel_per_poly;
          std::vector<Data> w_msg(ct_info.N, 0);

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
              ct_info.context_p->encode_coeffs_ringt(
                  w_msg));
        }
      }
    }
  };

  // helper function for parallelize the computation
  static void decode_par(
      const int Co, const int tile, CryptoInfo& ct_info,
      std::vector<::vector<Data>>& local_out_msg_ls,
      int co_start, int co_end, int thread_id,
      std::vector<BfvPlaintext>& out_mask) {
#ifdef LOG_LAYERWISE
    auto decode_timer =
        std::chrono::high_resolution_clock::now();
#endif

    for(int co = co_start; co < co_end; co++) {
      for(int ti = 0; ti < tile; ti++) {
        for(int tj = 0; tj < tile; tj++) {
          std::vector<Data> res =
              ct_info.context_par[thread_id]->decode_coeffs(
                  out_mask[Arr3DIdx(Co, tile, tile, co, ti,
                                    tj)]);
          local_out_msg_ls.push_back(res);
        }
      }
    }

#ifdef LOG_LAYERWISE
    if(thread_id == 0){
      auto temp =
          std::chrono::duration_cast<
              std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() -
              decode_timer)
              .count();
      std::cout << "parallel decode time in second: " << temp/1000.0
                << std::endl;
      Conv2dDEcodeParInMilliSec += temp;
    }
#endif
  }

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
          std::vector<Data> mask_msg(ct_info.N, 0);
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
  };

  /*
   * Forward the convolution operator.
   */
  Tensor<Data> forward(Tensor<Data>& input) {
    io->sync();
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

    // Input packing
    io->sync();
#ifdef LOG_LAYERWISE
    auto acpacktimestart = TIMER_TILL_NOW;
#endif

    if(party == sci::ALICE) {
      // pack the activation tensor
      ac_msg = pack_ac(input);
#ifdef MULT_THREAD_LINEAR
          std::vector<std::vector<BfvCiphertext>>
              ac_ct_thread(num_threads);
          auto packac_thread =
              [&ac_msg, this, &ac_ct_thread, &ioArr](
                  int thread_id, int start, int end) {
                vector<BfvPlaintext> ac_pt_thread;
                for(int i = start; i < end; i++) {
                  ac_pt_thread.push_back(
                      this->ct_info.context_par[thread_id]
                          ->encode_coeffs(
                              ac_msg[i],
                              this->ct_info.level));
                }
                for(int i = start; i < end; i++) {
                  vector<uint8_t> buffer(
                      this->ct_info.cipher_size, 0);
                  ioArr[thread_id]->recv_data(
                      static_cast<void*>(buffer.data()),
                      this->ct_info.cipher_size);
                  BfvCiphertext ct_buffer =
                      BfvCiphertext::deserialize(&buffer);
                  ac_ct_thread[thread_id].push_back(
                      move(ct_buffer));
                }
                for(int i = start; i < end; i++) {
                  this->ct_info.context_par[thread_id]
                      ->add_plain_inplace(ac_ct_thread[thread_id][i-start],
                                          ac_pt_thread[i-start]);
                }
              };

          std::vector<std::thread> threads(num_threads);
          size_t workloads = ac_msg.size() / num_threads;

          for(size_t i = 0; i < num_threads; i++) {
            size_t start = i * workloads;
            size_t end = (i == num_threads - 1)
                             ? ac_msg.size()
                             : (i + 1) * workloads;
            threads[i] =
                std::thread(packac_thread, i, start, end);
          }
          for(auto& t : threads) t.join();
          threads.clear();
          threads.resize(num_threads);

          for(int i = 0; i < num_threads; i++) {
            std::move(ac_ct_thread[i].begin(),
                      ac_ct_thread[i].end(),
                      std::back_inserter(ac_ct));
          }

#else
          for(int i = 0; i < ac_msg.size(); i++) {
            ac_pt.push_back(
                ct_info.context_p->encode_coeffs(
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
#endif
    } else {
      // for client
      ac_msg = pack_ac(input);
#ifdef MULT_THREAD_LINEAR
      std::vector<std::vector<BfvCiphertext>> ac_ct_thread(
          num_threads);
      auto packac_thread = [&ac_msg, this, &ac_ct_thread,
                            &ioArr](int thread_id,
                                    int start, int end) {
        for(int i = start; i < end; i++) {
          ac_ct_thread[thread_id].push_back(
              this->ct_info.context_par[thread_id]
                  ->encrypt_symmetric(
                      this->ct_info.context_par[thread_id]
                          ->encode_coeffs(
                              ac_msg[i],
                              this->ct_info.level)));

          ioArr[thread_id]->send_data(
              static_cast<void*>(ac_ct_thread[thread_id][i-start]
                                     .serialize(*(ct_info.param))
                                     .data()),
              this->ct_info.cipher_size);
        }
      };

      std::vector<std::thread> threads(num_threads);
      size_t workloads = ac_msg.size() / num_threads;

      for(size_t i = 0; i < num_threads; i++) {
        size_t start = i * workloads;
        size_t end = (i == num_threads - 1)
                         ? ac_msg.size()
                         : (i + 1) * workloads;
        threads[i] =
            std::thread(packac_thread, i, start, end);
      }

      for(auto& t : threads) t.join();

      threads.clear();
      threads.resize(num_threads);

#else
      for(int i = 0; i < ac_msg.size(); i++) {
        ac_ct.push_back(
            ct_info.context_p->encrypt_symmetric(
                ct_info.context_p->encode_coeffs(
                    ac_msg[i], ct_info.level)));
      }
      for(int i = 0; i < ac_msg.size(); i++) {
        io->send_data(
            static_cast<void*>(ac_ct[i].serialize(*ct_info.param).data()),
            ct_info.cipher_size);
      }
#endif
    }
  io->sync();
#ifdef LOG_LAYERWISE
    auto acpacktimeend = TIMER_TILL_NOW;
    Conv2dpackingacInMilliSec +=
        (acpacktimeend - acpacktimestart);
    std::cout << " Input packing takes: "
              << (acpacktimeend - acpacktimestart) / 1000.0
              << " s" << std::endl;
#endif

        // Perform computation
        if(party == sci::ALICE) {
      // vector<BfvCiphertext> y_ct;
#ifdef LOG_LAYERWISE
      auto computationstart = TIMER_TILL_NOW;
#endif
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            // vector<Data> all_zeros(8192, 0);
            // BfvPlaintext temp_buffer_1 =
            //     ct_info.context_p->encode_coeffs(
            //         all_zeros, ct_info.level);
            // BfvPlaintext temp_buffer_2 =
            //     ct_info.context_p->encode_coeffs(
            //         all_zeros, ct_info.level);
            // y_ct.push_back(
            //     ct_info.context_p->encrypt_asymmetric(
            //         temp_buffer_1));
            out_ct_ls.push_back(
                ct_info.context_p->new_ciphertext(1,
                    ct_info.level));
          }
        }
      }
#ifdef LOG_LAYERWISE
      auto comptationencode = TIMER_TILL_NOW; // Why is this encoding?
      std::cout << "Time in sec for encoding = "
                << ((comptationencode - computationstart) /
                    1000.0)
                << std::endl;
      Conv2dCoeffencodingInMilliSec +=
          ((comptationencode - computationstart));
#endif

#ifdef LOG_LAYERWISE
 auto kernelcomputationstart= TIMER_TILL_NOW;
#endif
      // std::string pth_to_kernel = pth;
      // FpgaProject fpga_project(pth_to_kernel);
//  std::cout << "DEBUG: a: " << ac_ct.size() << " w: " << weight_pt.size() << " out: " << out_ct_ls.size() << std::endl;
 vector<CxxVectorArgument> cxx_args = {
     {"a", &ac_ct},
     {"w", &weight_pt},
     // {"y", &y_ct},
     {"out", &out_ct_ls},
 };
 // auto computation_run_time = fpga_project.run(ct_info.context_p,
 //                                  cxx_args, true);
 auto computation_run_time = project->run(ct_info.context_p,
                                          cxx_args, true);

#ifdef LOG_LAYERWISE
      std::cout << "FPGA run time in second = "
                << static_cast<double>(
                       computation_run_time) /
                       (1000000000.0)
                << std::endl;
      Conv2dFpgaTimeInMilliSec +=
          static_cast<double>(computation_run_time) /
          (1000000.0);
      auto kernelcomputationend = TIMER_TILL_NOW;
      std::cout << "Time in sec for conv computation = "
                << ((kernelcomputationend -
                     kernelcomputationstart) /
                    1000.0)
                << std::endl;

      Conv2dCompTimeInMilliSec +=
          (kernelcomputationend - kernelcomputationstart);
#endif
    }
io->sync();
    /***** Generate the secret share *****/
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<Data> distrib(
        0, ct_info.prime_mod - 1);
#ifdef LOG_LAYERWISE
    auto maskstart = TIMER_TILL_NOW;
#endif
    if(party == sci::ALICE) {
      vector<vector<BfvPlaintext>> out_mask_thread(
          num_threads);
#ifdef MULT_THREAD_LINEAR

      auto mask_thread =
          [&out_ct_ls, &out_mask_thread, this, &distrib,
           &gen](int thread_id, int start, int end) {
            for(int co = start; co < end; co++) {
              for(int ti = 0; ti < tile; ti++) {
                for(int tj = 0; tj < tile; tj++) {
                  vector<Data> pos_mask(ct_info.N, 0);
                  vector<Data> neg_mask(ct_info.N, 0);
                  for(int i = 0; i < pos_mask.size(); i++) {
                    pos_mask[i] = distrib(gen);
                    neg_mask[i] =
                        ct_info.prime_mod - pos_mask[i];
                  }
                  BfvPlaintext temp_buffer_pos =
                      ct_info.context_par[thread_id]
                          ->encode_coeffs(pos_mask,
                                          ct_info.level);
                  BfvPlaintext temp_buffer_neg =
                      ct_info.context_par[thread_id]
                          ->encode_coeffs(neg_mask,
                                          ct_info.level);
                  ct_info.context_par[thread_id]
                      ->add_plain_inplace(
                          out_ct_ls[Arr3DIdx(Co, tile, tile,
                                             co, ti, tj)],
                          temp_buffer_neg);
                  out_mask_thread[thread_id].push_back(
                      move(temp_buffer_pos));
                }
              }
            }
            for(int co = start; co < end; co++) {
              for(int ti = 0; ti < tile; ti++) {
                for(int tj = 0; tj < tile; tj++) {
                  ioArr[thread_id]->send_data(
                      static_cast<void*>(
                          out_ct_ls[(Arr3DIdx(Co, tile,
                                              tile, co, ti,
                                              tj))]
                              .serialize(*(ct_info.param))
                              .data()),
                      this->ct_info.cipher_size);
                }
              }
            }
          };
      std::vector<std::thread> threads(num_threads);

      for(int i = 0; i < num_threads; ++i) {
        int startCo = i * Co / num_threads;
        size_t endCo = (i == num_threads - 1)
                         ? Co
                         : (i + 1) * Co/num_threads;
        threads[i] =
            std::thread(mask_thread,i, startCo, endCo);
      }

      for(auto& t : threads) {
        t.join();
      }
      for(int i = 0; i < num_threads; i++) {
        std::move(out_mask_thread[i].begin(),
                  out_mask_thread[i].end(),
                  std::back_inserter(out_mask));
      }
      threads.clear();
      threads.resize(num_threads);

#else
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            // TODO: needs more secure PRG
            vector<Data> pos_mask(ct_info.N, 0);
            vector<Data> neg_mask(ct_info.N, 0);
            for(int i = 0; i < pos_mask.size(); i++) {
              // pos_mask[i] = distrib(gen);
              // neg_mask[i] = ct_info.prime_mod - pos_mask[i];
              pos_mask[i]=0;
              neg_mask[i]=0;
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

#ifdef LOG_LAYERWISE
    auto send_start = TIMER_TILL_NOW;
#endif

      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            io->send_data(
                static_cast<void*>(
                    out_ct_ls[(Arr3DIdx(Co, tile, tile, co,
                                        ti, tj))]
                        .serialize(*ct_info.param)
                        .data()),
                ct_info.cipher_size);
          }
        }
      }
      
#ifdef LOG_LAYERWISE
    auto send_end = TIMER_TILL_NOW;
    Conv2dSendTimeInMilliSec += (send_end - send_start);
    std::cout << "Time in sec for sending = "
              << ((send_end - send_start) / 1000.0)
              << std::endl;
#endif

#endif
    } else {
#ifdef MULT_THREAD_LINEAR
      std::vector<std::vector<BfvCiphertext>> out_ct_ls_par(
          num_threads);
      auto mask_thread = [&out_ct_ls_par, &ioArr, this](
                             int thread_id, int start,
                             int end) {
        for(int i = start; i < end; i++) {
          for(int ti = 0; ti < tile; ti++) {
            for(int tj = 0; tj < tile; tj++) {
              vector<uint8_t> buffer(ct_info.cipher_size,
                                     0);
              ioArr[thread_id]->recv_data(
                  static_cast<void*>(buffer.data()),
                  ct_info.cipher_size);
              BfvCiphertext ct_buffer =
                  BfvCiphertext::deserialize(&buffer);
              out_ct_ls_par[thread_id].push_back(
                  move(ct_buffer));
            }
          }
        }
      };
      std::vector<std::thread> threads(num_threads);
      for(int i = 0; i < num_threads; ++i) {
        int startCo = i * Co / num_threads;
        size_t endCo = (i == num_threads - 1)
                           ? Co
                           : (i + 1) * Co / num_threads;
        threads[i] =
            std::thread(mask_thread, i, startCo, endCo);
            }

            for(auto& t : threads) {
              t.join();
            }
            for(int i = 0; i < num_threads; i++) {
              std::move(out_ct_ls_par[i].begin(),
                        out_ct_ls_par[i].end(),
                        std::back_inserter(out_ct_ls));
            }

#else
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
      #endif
    }
io->sync();
#ifdef LOG_LAYERWISE
    auto maskend = TIMER_TILL_NOW;
    Conv2dMaskTimeInMilliSec += (maskend - maskstart);
    std::cout << "Time in sec for mask = "
              << ((maskend - maskstart) / 1000.0)
              << std::endl;
    auto decryptandunpackstart = TIMER_TILL_NOW;
#endif
    /***** decrypt and unpack *****/
    if(party == sci::ALICE) {
#ifdef MULT_THREAD_LINEAR
      std::vector<std::thread> threads(num_threads);
      std::vector<std::vector<std::vector<Data>>>
          out_msg_ls_par(num_threads);
      int co_per_thread = Co / num_threads;
      for(int i = 0; i < num_threads; i++) {
        int co_start = i * co_per_thread;
        int co_end = (i + 1) * co_per_thread;
        if(i == num_threads - 1) co_end = Co;
        threads[i] = std::thread(
            decode_par, Co, tile, std::ref(ct_info),
            std::ref(out_msg_ls_par[i]), co_start, co_end,
            i, std::ref(out_mask));
      }

      for(auto& t : threads) {
        t.join();
      }

      for(int i = 0; i < num_threads; i++) {
        out_msg_ls.insert(out_msg_ls.end(),
                          out_msg_ls_par[i].begin(),
                          out_msg_ls_par[i].end());
      }

#else
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            vector<Data> res =
                ct_info.context_p->decode_coeffs(
                    out_mask[Arr3DIdx(Co, tile, tile, co,
                                      ti, tj)]);
            out_msg_ls.push_back(res);
          }
        }
      }
#endif
    } else {
#ifdef MULT_THREAD_LINEAR
      std::vector<std::thread> threads(num_threads);
      std::vector<std::vector<std::vector<Data>>>
          out_msg_ls_par(num_threads);
      auto decode_thread = [&out_msg_ls_par, &out_ct_ls,
                            &ioArr,
                            this](int thread_id, int start,
                                  int end) {
        for(int co = start; co < end; co++) {
          for(int ti = 0; ti < tile; ti++) {
            for(int tj = 0; tj < tile; tj++) {
              vector<Data> res =
                  ct_info.context_par[thread_id]
                      ->decode_coeffs(
                          ct_info.context_par[thread_id]
                              ->decrypt(out_ct_ls[Arr3DIdx(
                                  Co, tile, tile, co, ti,
                                  tj)]));
              out_msg_ls_par[thread_id].push_back(res);
            }
          }
        }
      };
      int co_per_thread = Co / num_threads;
      for(int i = 0; i < num_threads; i++) {
        int co_start = i * co_per_thread;
        int co_end = (i + 1) * co_per_thread;
        if(i == num_threads - 1) co_end = Co;
        threads[i] = std::thread(
            decode_thread, i, co_start, co_end);
        }

        for(auto& t : threads) {
          t.join();
        }

        for(int i = 0; i < num_threads; i++) {
          out_msg_ls.insert(out_msg_ls.end(),
                            out_msg_ls_par[i].begin(),
                            out_msg_ls_par[i].end());
        }

#else
      for(int co = 0; co < Co; co++) {
        for(int ti = 0; ti < tile; ti++) {
          for(int tj = 0; tj < tile; tj++) {
            vector<Data> res =
                ct_info.context_p->decode_coeffs(
                    ct_info.context_p->decrypt(
                        out_ct_ls[Arr3DIdx(Co, tile, tile,
                                           co, ti, tj)]));
            out_msg_ls.push_back(res);
          }
        }
      }
#endif
    }
        Tensor<Data> restensor = depack_res(out_msg_ls);
        io->sync();
#ifdef LOG_LAYERWISE
        auto decryptandunpackend = TIMER_TILL_NOW;
        Conv2dDecryptUnpackInMilliSec +=
            (decryptandunpackend - decryptandunpackstart);
        std::cout << "Time in sec for decrypt and unpack = "
                  << ((decryptandunpackend -
                       decryptandunpackstart) /
                      1000.0)
                  << std::endl;
#endif
#ifdef LOG_LAYERWISE
        auto temp = TIMER_TILL_NOW;
        Conv2dCoeffTimeInMilliSec += temp;
        std::cout << "Time in sec for current conv = "
                  << (temp / 1000.0) << std::endl;
        uint64_t curComm;
        FIND_ALL_IO_TILL_NOW(curComm);
        Conv2dCoeffCommSent += curComm;
#endif

        return restensor;
      };
  };


template <typename Data>
class Conv2dNest : public Module<Data> {
 public:
  Tensor<Data> weight;
  int Co, Ci, c_o, c_i, C, Hi, Wi, H, W, k, s, p, Ho, Wo;
  std::vector<BfvPlaintextMul> weight_pt;
  CryptoInfo ct_info;
  std::shared_ptr<FpgaProject> project;

  Conv2dNest(){};

  Conv2dNest(Tensor<Data>& weight, int Co, int Ci, int Hi, int Wi, int k, int s, int p, CryptoInfo crypto_info, std::shared_ptr<FpgaProject> project)
      : Module<Data>(),
        weight(weight),
        Co(Co),
        Ci(Ci),
        Hi(Hi),
        Wi(Wi),
        k(k),
        s(s),
        p(p),
        ct_info(crypto_info), 
        project(project) {

    H = Hi + 2 * p;
    W = Wi + 2 * p;
    Ho = (H - k) / s + 1;
    Wo = (W - k) / s + 1;
    H--;
    W--;
    for (int i = 0; i < 5; i++){
      // To check for the largest power of 2 less than or equal to H and W
      H |= H >> (1 << i);
      W |= W >> (1 << i);
    }
    H++;
    W++;
    C = ct_info.N / (2 * H * W);
    Co /= 2; // Co should be even
    c_i = Ci / C + (Ci % C != 0);
    c_o = Co / C + (Co % C != 0); 

    uint64_t modulus = ct_info.prime_mod;
    size_t degree = H * W;
    intel::hexl::NTT ntt(degree, modulus);

    Tensor<Data> weight1(TensorShape(4, {k, k, Ci, Co}));
    Tensor<Data> weight2(TensorShape(4, {k, k, Ci, Co}));
    for (int i = 0; i < k; i++){
      for (int j = 0; j < k; j++){
        for (int m = 0; m < Ci; m++){
          for (int n = 0; n < Co; n++){
            weight1.cached_data[m * k * k * Co + n * k * k + i * k + j] = weight.cached_data[n * k * k * Ci + m * k * k + i * k + j];
            weight2.cached_data[m * k * k * Co + n * k * k + i * k + j] = weight.cached_data[(n + Co) * k * k * Ci + m * k * k + i * k + j]; // to be confirmed
          }
        }
      }
    }

    if (party == sci::ALICE){
      int inrot = std::sqrt(C);
      if (inrot != std::floor(inrot)){
        inrot *= std::sqrt(2);
      }
      int os = (k - 1) * (W + 1);
      for (int i = 0; i < c_i; i++){
        for (int j = 0; j < c_o; j++){
          for (int l = 0; l < C; l++){
            std::vector<uint64_t> w_msg(ct_info.N, 0);
            for (int m = 0; m < C; m++){
              int ci = i * C + (m + (inrot - 1 - l % inrot)) % C;
              int co = j * C + (3 * C - m - l - (inrot - 1 - l % inrot) - 1) % C;
              if (ci < Ci && co < Co){
                for (int n = 0; n < k * k; n++){
                  w_msg[m * H * W + os - (n / k) * W - (n % k)] = weight1.cached_data[ci * k * k * Co + co * k * k + n];
                  w_msg[(m + C) * H * W + os - (n / k) * W - (n % k)] = weight2.cached_data[ci * k * k * Co + co * k * k + n]; 
                }
              }
            }
            for (int m = 0; m < 2 * C; m++){ // somewhat hideous and redundant, but it works
              std::vector<uint64_t> tmp(H * W, 0);
              for (int n = 0; n < H * W; n++){
                tmp[n] = w_msg[m * H * W + n];
              }
              ntt.ComputeForward(tmp.data(), tmp.data(), 1, 1);
              for (int n = 0; n < H * W; n++){
                w_msg[m * H * W + n] = tmp[n];
              }
            }
            weight_pt.push_back(ct_info.context_p->encode_mul(w_msg, ct_info.level));
          }
        }
      }
    }
  }
    
  vector<vector<Data>> pack_ac(Tensor<Data>& act) {
    std::vector<vector<Data>> packed_ac;
    uint64_t modulus = ct_info.prime_mod;
    size_t degree = H * W;
    intel::hexl::NTT ntt(degree, modulus);

    for (int i = 0; i < c_i; i++){
      std::vector<uint64_t> ac_msg(ct_info.N, 0);
      for (int j = 0; j < C; j++){
        if (i * C + j < Ci){
          for (int m = 0; m < H; m++){
            for (int n = 0; n < W; n++){
              if (m >= p && m < p + Hi && n >= p && n < p + Wi){
                ac_msg[j * H * W + m * W + n] = act.cached_data[(i * C + j) * Hi * Wi + (m - p) * Wi + (n - p)];
                ac_msg[(j + C) * H * W + m * W + n] = act.cached_data[(i * C + j) * Hi * Wi + (m - p) * Wi + (n - p)];
              }
            }
          }
        }
      }

      for (int m = 0; m < 2 * C; m++){
        std::vector<uint64_t> tmp(H * W, 0);
        for (int n = 0; n < H * W; n++){
          tmp[n] = ac_msg[m * H * W + n];
        }
        ntt.ComputeForward(tmp.data(), tmp.data(), 1, 1);
        for (int n = 0; n < H * W; n++){
          ac_msg[m * H * W + n] = tmp[n];
        }
      }
      packed_ac.push_back(ac_msg);
    }

    return packed_ac;
  };

  Tensor<Data> depack_res(vector<vector<Data>>& out_msg) {
    uint64_t modulus = ct_info.prime_mod;
    size_t degree = H * W;

// io->sync();
// #ifdef LOG_LAYERWISE
//     auto depacktimestart = TIMER_TILL_NOW;
// #endif

    intel::hexl::NTT ntt(degree, modulus);

    // for (int i = 0; i < c_o; i++){
    //   for (int m = 0; m < 2 * C; m++){
    //     std::vector<uint64_t> tmp(H * W, 0);
    //     for (int n = 0; n < H * W; n++){
    //       tmp[n] = out_msg[i][m * H * W + n];
    //     }
    //     ntt.ComputeInverse(tmp.data(), tmp.data(), 1, 1);
    //     for (int n = 0; n < H * W; n++){
    //       out_msg[i][m * H * W + n] = tmp[n];
    //     }
    //   } 
    // }

    Co *= 2;
    Tensor<Data> out(TensorShape(3, {Co, Ho, Wo}));
    for (int i = 0; i < Co; i++){
      for (int j = 0; j < Ho; j++){
        for (int l = 0; l < Wo; l++){
          int index = s * j * W + s * l + (k - 1) * (W + 1);
          if (i < Co / 2){
            out.cached_data[i * Ho * Wo + j * Wo + l] = out_msg[i / C][((C * Co - i) % C) * H * W + index];
          }
          else{
            out.cached_data[i * Ho * Wo + j * Wo + l] = out_msg[(i - Co / 2) / C][(C + (C * Co - i + Co / 2) % C) * H * W + index];
          }
        }
      }
    }

// #ifdef LOG_LAYERWISE
//     auto depacktimeend = TIMER_TILL_NOW;
//     Conv2dpackingacInMilliSec +=
//         (depacktimeend - depacktimestart);
//     std::cout << "NTT and depacking takes: "
//               << (depacktimeend - depacktimestart) / 1000.0
//               << " s" << std::endl;
// #endif

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
    vector<BfvCiphertext> int_ct;
    vector<BfvCiphertext> out_ct;
    vector<BfvPlaintext> out_share;
    vector<vector<Data>> out_msg;

  io->sync();
#ifdef LOG_LAYERWISE
    auto acpacktimestart = TIMER_TILL_NOW;
#endif

    Co /= 2;
    if (party == sci::ALICE){
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
        io->send_data(static_cast<void*>(ac_ct[i].serialize(*ct_info.param).data()), ct_info.cipher_size);
      }
    }

  io->sync();
#ifdef LOG_LAYERWISE
    auto acpacktimeend = TIMER_TILL_NOW;
    Conv2dpackingacInMilliSec +=
        (acpacktimeend - acpacktimestart);
    std::cout << " Input packing takes: "
              << (acpacktimeend - acpacktimestart) / 1000.0
              << " s" << std::endl;
#endif

#ifdef LOG_LAYERWISE
 auto kernelcomputationstart= TIMER_TILL_NOW;
#endif

    if (party == sci::ALICE){
      for (int i = 0; i < c_o; i++){
        out_ct.push_back(std::move(ct_info.context_p->new_ciphertext(1, ct_info.level)));
      }
      vector<CxxVectorArgument> cxx_args = {{"w", &weight_pt}, {"ac", &ac_ct}, {"y", &out_ct} };
      auto computation_run_time = project->run(ct_info.context_p, cxx_args, true);

#ifdef LOG_LAYERWISE
      std::cout << "FPGA run time in second = "
                << static_cast<double>(
                       computation_run_time) /
                       (1000000000.0)
                << std::endl;
      Conv2dFpgaTimeInMilliSec +=
          static_cast<double>(computation_run_time) /
          (1000000.0);
      auto kernelcomputationend = TIMER_TILL_NOW;
      std::cout << "Time in sec for conv computation = "
                << ((kernelcomputationend -
                     kernelcomputationstart) /
                    1000.0)
                << std::endl;

      Conv2dCompTimeInMilliSec +=
          (kernelcomputationend - kernelcomputationstart);
#endif

    }
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> distrib(0, ct_info.prime_mod - 1);

#ifdef LOG_LAYERWISE
    auto maskstart = TIMER_TILL_NOW;
#endif

    if (party == sci::ALICE){
      for (int i = 0; i < c_o; i++){
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
      }

#ifdef LOG_LAYERWISE
    auto send_start = TIMER_TILL_NOW;
#endif

      for (int i = 0; i < c_o; i++){
        io->send_data(static_cast<void*>(out_ct[i].serialize(*ct_info.param).data()), ct_info.cipher_size);
      }

#ifdef LOG_LAYERWISE
    auto send_end = TIMER_TILL_NOW;
    Conv2dSendTimeInMilliSec += (send_end - send_start);
    std::cout << "Time in sec for sending = "
              << ((send_end - send_start) / 1000.0)
              << std::endl;
#endif

    }
    else {
      for(int i = 0; i < c_o; i++) {
        vector<uint8_t> buffer(ct_info.cipher_size, 0);
        io->recv_data(static_cast<void*>(buffer.data()), ct_info.cipher_size);
        BfvCiphertext ct_buffer = BfvCiphertext::deserialize(&buffer);
        out_ct.push_back(move(ct_buffer));
      }
    }

io->sync();
#ifdef LOG_LAYERWISE
    auto maskend = TIMER_TILL_NOW;
    Conv2dMaskTimeInMilliSec += (maskend - maskstart);
    std::cout << "Time in sec for mask = "
              << ((maskend - maskstart) / 1000.0)
              << std::endl;
    auto decryptandunpackstart = TIMER_TILL_NOW;
#endif

    if(party == sci::ALICE) {
      for(int i = 0; i < c_o; i++) {
        vector<uint64_t> res = ct_info.context_p->decode(out_share[i]);
        out_msg.push_back(res);
      }
    } else {
      for(int i = 0; i < c_o; i++) {
        vector<uint64_t> res = ct_info.context_p->decode(ct_info.context_p->decrypt(out_ct[i]));
        out_msg.push_back(res);
      }
    }

        io->sync();
#ifdef LOG_LAYERWISE
        auto decryptandunpackend = TIMER_TILL_NOW;
        Conv2dDecryptUnpackInMilliSec +=
            (decryptandunpackend - decryptandunpackstart);
        std::cout << "Time in sec for decrypt and unpack = "
                  << ((decryptandunpackend -
                       decryptandunpackstart) /
                      1000.0)
                  << std::endl;
#endif

#ifdef LOG_LAYERWISE
        auto temp = TIMER_TILL_NOW;
        Conv2dCoeffTimeInMilliSec += temp;
        std::cout << "Time in sec for current conv = "
                  << (temp / 1000.0) << std::endl;
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
class ResNet18CoeffFPGA : public Module<Data> {
 public:
  CryptoInfo ct_info;
  Conv2dCoeffFPGA<Data> conv0;
  Conv2dNest<Data> conv1, conv2, conv3, conv4,
      conv5, conv6, conv7, conv8, conv9, conv10, conv11,
      conv12, conv13, conv14, conv15, conv16, conv17,
      conv18, conv19;

  ResNet18CoeffFPGA(std::vector<std::string> weight_pth,
                    CryptoInfo ct_info,
                    // std::vector<std::string> fpga_pth
                    std::vector<std::shared_ptr<FpgaProject>> project_ls
                    )
      : ct_info(ct_info) {
    // Initial block
    Tensor<Data> weight0(TensorShape(4, {64, 3, 7, 7}));
    weight0.load(weight_pth[0]);
    conv0 =
        Conv2dCoeffFPGA<Data>(weight0, 64, 3, 224, 224, 7,
                         2, 3, ct_info, project_ls[0]);
    // First Block
    Tensor<Data> weight1(TensorShape(4, {64, 64, 3, 3}));
    weight1.load(weight_pth[1]);
    conv1 =
        Conv2dNest<Data>(weight1, 64, 64, 56, 56, 3, 1,
                              1, ct_info,  project_ls[1]);

    Tensor<Data> weight2(TensorShape(4, {64, 64, 3, 3}));
    weight2.load(weight_pth[2]);
    conv2 =
        Conv2dNest<Data>(weight2, 64, 64, 56, 56, 3, 1,
                              1, ct_info, project_ls[2]);
    Tensor<Data> weight3(TensorShape(4, {64, 64, 3, 3}));
    weight3.load(weight_pth[3]);
    conv3 =
        Conv2dNest<Data>(weight3, 64, 64, 56, 56, 3, 1,
                              1, ct_info, project_ls[3]);
    Tensor<Data> weight4(TensorShape(4, {64, 64, 3, 3}));
    weight4.load(weight_pth[4]);
    conv4 =
        Conv2dNest<Data>(weight4, 64, 64, 56, 56, 3, 1,
                              1, ct_info,project_ls[4]);

    // Second Block
    // Block 1
    Tensor<Data> weight5(TensorShape(4, {128, 64, 3, 3}));
    weight5.load(weight_pth[5]);
    conv5 =
        Conv2dNest<Data>(weight5, 128, 64, 56, 56, 3,
                              2, 1, ct_info, project_ls[5]);
    Tensor<Data> weight6(TensorShape(4, {128, 128, 3, 3}));
    weight6.load(weight_pth[6]);
    conv6 =
        Conv2dNest<Data>(weight6, 128, 128, 28, 28, 3,
                              1, 1, ct_info, project_ls[6]);
    Tensor<Data> weight7(
        TensorShape(4, {128, 64, 1, 1}));  // short cut
    weight7.load(weight_pth[7]);
    conv7 =
        Conv2dNest<Data>(weight7, 128, 64, 56, 56, 1,
                              2, 0, ct_info, project_ls[7]);

    // Block 2
    Tensor<Data> weight8(TensorShape(4, {128, 128, 3, 3}));
    weight8.load(weight_pth[8]);
    conv8 =
        Conv2dNest<Data>(weight8, 128, 128, 28, 28, 3,
                              1, 1, ct_info, project_ls[8]);
    Tensor<Data> weight9(TensorShape(4, {128, 128, 3, 3}));
    weight9.load(weight_pth[9]);
    conv9 =
        Conv2dNest<Data>(weight9, 128, 128, 28, 28, 3,
                              1, 1, ct_info, project_ls[9]);

    // Third Block
    // Block 1
    Tensor<Data> weight10(TensorShape(4, {256, 128, 3, 3}));
    weight10.load(weight_pth[10]);
    conv10 =
        Conv2dNest<Data>(weight10, 256, 128, 28, 28, 3,
                              2, 1, ct_info, project_ls[10]);
    Tensor<Data> weight11(TensorShape(4, {256, 256, 3, 3}));
    weight11.load(weight_pth[11]);
    conv11 =
        Conv2dNest<Data>(weight11, 256, 256, 14, 14, 3,
                              1, 1, ct_info,project_ls[11]);
    Tensor<Data> weight12(
        TensorShape(4, {256, 128, 3, 3}));  // short cut
    weight12.load(weight_pth[12]);
    conv12 =
        Conv2dNest<Data>(weight12, 256, 128, 28, 28, 1,
                              2, 0, ct_info, project_ls[12]);
    // Block 2
    Tensor<Data> weight13(TensorShape(4, {256, 256, 3, 3}));
    weight13.load(weight_pth[13]);
    conv13 =
        Conv2dNest<Data>(weight13, 256, 256, 14, 14, 3,
                              1, 1, ct_info,project_ls[13]);
    Tensor<Data> weight14(TensorShape(4, {256, 256, 3, 3}));
    weight14.load(weight_pth[14]);
    conv14 =
        Conv2dNest<Data>(weight14, 256, 256, 14, 14, 3,
                              1, 1, ct_info, project_ls[14]);

    // Fourth Block
    // Block 1
    Tensor<Data> weight15(TensorShape(4, {512, 256, 3, 3}));
    weight15.load(weight_pth[15]);
    conv15 =
        Conv2dNest<Data>(weight15, 512, 256, 14, 14, 3,
                              2, 1, ct_info, project_ls[15]);
    Tensor<Data> weight16(TensorShape(4, {512, 512, 3, 3}));
    weight16.load(weight_pth[16]);
    conv16 =
        Conv2dNest<Data>(weight16, 512, 512, 7, 7, 3,
                              1, 1, ct_info, project_ls[16]);
    Tensor<Data> weight17(
        TensorShape(4, {512, 256, 3, 3}));  // short cut
    weight17.load(weight_pth[17]);
    conv17 =
        Conv2dNest<Data>(weight17, 512, 256, 14, 14, 1,
                              2, 0, ct_info, project_ls[17]);

    // Block 2
    Tensor<Data> weight18(TensorShape(4, {512, 512, 3, 3}));
    weight18.load(weight_pth[18]);
    conv18 =
        Conv2dNest<Data>(weight18, 512, 512, 7, 7, 3,
                              1, 1, ct_info, project_ls[18]);
    Tensor<Data> weight19(TensorShape(4, {512, 512, 3, 3}));
    weight19.load(weight_pth[19]);
    conv19 =
        Conv2dNest<Data>(weight19, 512, 512, 7, 7, 3,
                              1, 1, ct_info, project_ls[19]);
  }

  Tensor<Data> forward(Tensor<Data> input, int sf) {
    std::cout << "----- Initial Block -----" << std::endl;

    // Block 1
    Tensor<Data> ac1 = conv0.forward(input);
    Tensor<Data> ac2 = Relu<Data>(ac1, sf, 1);
    Tensor<Data> ac3 = MaxPool(64, 112, 112, 3, 1, 2, ac2);

    std::cout << "----- Block 2-1 -----" << std::endl;

    // Block 2-1
    Tensor<Data> ac4 = conv1.forward(ac3);
    Tensor<Data> ac5 = Relu<Data>(ac4, sf, 1);
    Tensor<Data> ac6 = conv2.forward(ac5);
    Tensor<Data> ac7 = Relu<Data>(ac6 + ac3, sf, 1);

    std::cout << "----- Block 2-2 -----" << std::endl;

    // Block 2-2
    Tensor<Data> ac8 = conv3.forward(ac7);
    Tensor<Data> ac9 = Relu<Data>(ac8, sf, 1);
    Tensor<Data> ac10 = conv4.forward(ac9);
    Tensor<Data> ac11 = Relu<Data>(ac10 + ac7, sf, 1);

    std::cout << "----- Block 3-1 -----" << std::endl;

    // Block 3-1
    Tensor<Data> ac12 = conv5.forward(ac11);
    Tensor<Data> ac13 = Relu<Data>(ac12, sf, 1);
    Tensor<Data> ac14 = conv6.forward(ac13);
    std::cout << "conv6 completd" << std::endl;
    // short cut
    Tensor<Data> ac15 =
        Relu<Data>(ac14 + conv7.forward(ac11), sf, 1);

    std::cout << "----- Block 3-2 -----" << std::endl;

    // Block 3-2
    Tensor<Data> ac16 = conv8.forward(ac15);
    Tensor<Data> ac17 = Relu<Data>(ac16, sf, 1);
    Tensor<Data> ac18 = conv9.forward(ac17);
    Tensor<Data> ac19 = Relu<Data>(ac18 + ac15, sf, 1);
    std::cout << "----- Block 4-1 -----" << std::endl;

    // Block 4-1
    Tensor<Data> ac20 = conv10.forward(ac19);
    Tensor<Data> ac21 = Relu<Data>(ac20, sf, 1);
    Tensor<Data> ac22 = conv11.forward(ac21);
    // short cut
    Tensor<Data> ac23 =
        Relu<Data>(ac22 + conv12.forward(ac19), sf, 1);
    std::cout << "----- Block 4-2 -----" << std::endl;

    // Block 4-2
    Tensor<Data> ac24 = conv13.forward(ac23);
    Tensor<Data> ac25 = Relu<Data>(ac24, sf, 1);
    Tensor<Data> ac26 = conv14.forward(ac25);
    Tensor<Data> ac27 = Relu<Data>(ac26 + ac23, sf, 1);
    std::cout << "----- Block 5-1 -----" << std::endl;

    // Block 5-1
    Tensor<Data> ac28 = conv15.forward(ac27);
    Tensor<Data> ac29 = Relu<Data>(ac28, sf, 1);
    Tensor<Data> ac30 = conv16.forward(ac29);
    // short cut
    Tensor<Data> ac31 =
        Relu<Data>(ac30 + conv17.forward(ac27), sf, 1);
    std::cout << "----- Block 5-2 -----" << std::endl;

    // Block 5-2
    Tensor<Data> ac32 = conv18.forward(ac31);
    Tensor<Data> ac33 = Relu<Data>(ac32, sf, 1);
    Tensor<Data> ac34 = conv19.forward(ac33);
    Tensor<Data> ac35 = Relu<Data>(ac34 + ac31, sf, 1);
    return ac35;
  }
};
#endif  // MODULE_H