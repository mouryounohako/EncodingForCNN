#include <inttypes.h>

#include <random>
#include <memory>

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

template <typename Data>
class Conv2dPlain : public Module<Data> {
 public:
  /* Tensor info */
  Tensor<Data> weight;
  int Co, Ci, H, W, k, s, p, Ho, Wo;

  Conv2dPlain(){};

  Conv2dPlain(Tensor<Data>& weight, int Co, int Ci,
                  int H, int W, int k, int s, int p)
      : Module<Data>(),
        weight(weight),
        Co(Co),
        Ci(Ci),
        H(H),
        W(W),
        k(k),
        s(s),
        p(p)
        {
        Ho = (H + 2 * p - k) / s + 1;
        Wo = (W + 2 * p - k) / s + 1;
  }

  vector<vector<Data>> pack_ac(Tensor<Data>& unpadded_ac) {
    return 0;
  };

  Tensor<Data> depack_res(vector<vector<Data>>& res_poly) {
    return 0;
  };

  /*
   * Forward the convolution operator.
   */
  Tensor<Data> forward(Tensor<Data>& input) {
    Tensor<Data> output(TensorShape(3, {Co, Ho, Wo})); // {a, b, c} means (i, j, k) => (i * b * c + j * c + k)
    // weight = Co * Ci * kh * kw, input = Ci * H * W
    // Ci * (H + 2 * p) * (W + 2 * p), Hi + p > H = kh + s * Ho >= p
    for (int i = 0; i < Co; i++){
        for (int j = 0; j < Ho; j++){
            for (int l = 0; l < Wo; l++){
                output.cached_data[i * Ho * Wo + j * Wo + l] = 0;
                for (int m = 0; m < k; m++){
                    for (int n = 0; n < k; n++){
                        for (int o = 0; o < Ci; o++){
                            if (n + s * j >= p && n + s * j < H + p && m + s * l >= p && m + s * l < W + p){
                                output.cached_data[i * Ho * Wo + j * Wo + l] += input.cached_data[o * H * W + (n + s * j - p) * W + (m + s * l - p)] * weight.cached_data[i * k * k * Ci + o * k * k + n * k + m];
                            }
                        }
                    }
                }
            }
        }
    }
    return output;
  };
};

template <typename Data>
Tensor<Data> ReLu(Tensor<Data> ac){
    for (int i = 0; i < ac.cached_data.size(); i++){
        if (ac.cached_data[i] < 0){
            ac.cached_data[i] = 0;
        }
    }

    return ac;
}

template <typename Data>
Tensor<Data> Maxpool(int C, int H, int W, int k, int p, int s, Tensor<Data> ac){
    int Ho = (H + 2 * p - k) / s + 1;
    int Wo = (W + 2 * p - k) / s + 1;
    Tensor<Data> output(TensorShape(3, {C, Ho, Wo}));
    for (int c = 0; c < C; c++){
        for (int h = 0; h < Ho; h++){
            for (int w = 0; w < Wo; w++){
                output.cached_data[c * Ho * Wo + h * Wo + w] = 0;
                for (int m = 0; m < k; m++){
                    for (int n = 0; n < k; n++){
                        if (n + s * h >= p && n + s * h < H + p && m + s * w >= p && m + s * w < W + p){
                            if (output.cached_data[c * Ho * Wo + h * Wo + w] < ac.cached_data[c * H * W + (n + s * h - p) * W + (m + s * w - p)]){
                                output.cached_data[c * Ho * Wo + h * Wo + w] = ac.cached_data[c * H * W + (n + s * h - p) * W + (m + s * w - p)];
                            }
                        } 
                    }
                }
            }
        }
    }
    
    return output;
}

/*
 * ResNet18Network in plaintext
 * @param Data: the data type of the underlying data, like
 * uint64_t.
 */
template <typename Data>
class ResNet18Plain : public Module<Data> {
 public:
  Conv2dPlain<Data> conv0, conv1, conv2, conv3, conv4,
      conv5, conv6, conv7, conv8, conv9, conv10, conv11,
      conv12, conv13, conv14, conv15, conv16, conv17,
      conv18, conv19;

  ResNet18Plain(std::vector<std::string> weight_pth){
    // Initial block
    Tensor<Data> weight0(TensorShape(4, {64, 3, 7, 7}));
    weight0.load(weight_pth[0]);
    conv0 =
        Conv2dPlain<Data>(weight0, 64, 3, 224, 224, 7, 2, 3);
    // First Block
    Tensor<Data> weight1(TensorShape(4, {64, 64, 3, 3}));
    weight1.load(weight_pth[1]);
    conv1 =
        Conv2dPlain<Data>(weight1, 64, 64, 56, 56, 3, 1, 1);

    Tensor<Data> weight2(TensorShape(4, {64, 64, 3, 3}));
    weight2.load(weight_pth[2]);
    conv2 =
        Conv2dPlain<Data>(weight2, 64, 64, 56, 56, 3, 1, 1);
    Tensor<Data> weight3(TensorShape(4, {64, 64, 3, 3}));
    weight3.load(weight_pth[3]);
    conv3 =
        Conv2dPlain<Data>(weight3, 64, 64, 56, 56, 3, 1, 1);
    Tensor<Data> weight4(TensorShape(4, {64, 64, 3, 3}));
    weight4.load(weight_pth[4]);
    conv4 =
        Conv2dPlain<Data>(weight4, 64, 64, 56, 56, 3, 1, 1);

    // Second Block
    // Block 1
    Tensor<Data> weight5(TensorShape(4, {128, 64, 3, 3}));
    weight5.load(weight_pth[5]);
    conv5 =
        Conv2dPlain<Data>(weight5, 128, 64, 56, 56, 3, 2, 1);
    Tensor<Data> weight6(TensorShape(4, {128, 128, 3, 3}));
    weight6.load(weight_pth[6]);
    conv6 =
        Conv2dPlain<Data>(weight6, 128, 128, 28, 28, 3, 1, 1);
    Tensor<Data> weight7(
        TensorShape(4, {128, 64, 1, 1}));  // short cut
    weight7.load(weight_pth[7]);
    conv7 =
        Conv2dPlain<Data>(weight7, 128, 64, 56, 56, 1, 2, 0);

    // Block 2
    Tensor<Data> weight8(TensorShape(4, {128, 128, 3, 3}));
    weight8.load(weight_pth[8]);
    conv8 =
        Conv2dPlain<Data>(weight8, 128, 128, 28, 28, 3, 1, 1);
    Tensor<Data> weight9(TensorShape(4, {128, 128, 3, 3}));
    weight9.load(weight_pth[9]);
    conv9 =
        Conv2dPlain<Data>(weight9, 128, 128, 28, 28, 3, 1, 1);

    // Third Block
    // Block 1
    Tensor<Data> weight10(TensorShape(4, {256, 128, 3, 3}));
    weight10.load(weight_pth[10]);
    conv10 =
        Conv2dPlain<Data>(weight10, 256, 128, 28, 28, 3, 2, 1);
    Tensor<Data> weight11(TensorShape(4, {256, 256, 3, 3}));
    weight11.load(weight_pth[11]);
    conv11 =
        Conv2dPlain<Data>(weight11, 256, 256, 14, 14, 3, 1, 1);
    Tensor<Data> weight12(
        TensorShape(4, {256, 128, 3, 3}));  // short cut
    weight12.load(weight_pth[12]);
    conv12 =
        Conv2dPlain<Data>(weight12, 256, 128, 28, 28, 1, 2, 0);
    // Block 2
    Tensor<Data> weight13(TensorShape(4, {256, 256, 3, 3}));
    weight13.load(weight_pth[13]);
    conv13 =
        Conv2dPlain<Data>(weight13, 256, 256, 14, 14, 3, 1, 1);
    Tensor<Data> weight14(TensorShape(4, {256, 256, 3, 3}));
    weight14.load(weight_pth[14]);
    conv14 =
        Conv2dPlain<Data>(weight14, 256, 256, 14, 14, 3, 1, 1);

    // Fourth Block
    // Block 1
    Tensor<Data> weight15(TensorShape(4, {512, 256, 3, 3}));
    weight15.load(weight_pth[15]);
    conv15 =
        Conv2dPlain<Data>(weight15, 512, 256, 14, 14, 3, 2, 1);
    Tensor<Data> weight16(TensorShape(4, {512, 512, 3, 3}));
    weight16.load(weight_pth[16]);
    conv16 =
        Conv2dPlain<Data>(weight16, 512, 512, 7, 7, 3, 1, 1);
    Tensor<Data> weight17(
        TensorShape(4, {512, 256, 3, 3}));  // short cut
    weight17.load(weight_pth[17]);
    conv17 =
        Conv2dPlain<Data>(weight17, 512, 256, 14, 14, 1, 2, 0);

    // Block 2
    Tensor<Data> weight18(TensorShape(4, {512, 512, 3, 3}));
    weight18.load(weight_pth[18]);
    conv18 =
        Conv2dPlain<Data>(weight18, 512, 512, 7, 7, 3, 1, 1);
    Tensor<Data> weight19(TensorShape(4, {512, 512, 3, 3}));
    weight19.load(weight_pth[19]);
    conv19 =
        Conv2dPlain<Data>(weight19, 512, 512, 7, 7, 3, 1, 1);
  }

  Tensor<Data> forward(Tensor<Data> input, int sf) {
    std::cout << "----- Initial Block -----" << std::endl;

    // Block 1
    Tensor<Data> ac1 = conv0.forward(input);
    Tensor<Data> ac2 = ReLu<Data>(ac1);
    Tensor<Data> ac3 = Maxpool(64, 112, 112, 3, 1, 2, ac2);

    std::cout << "----- Block 2-1 -----" << std::endl;

    // Block 2-1
    Tensor<Data> ac4 = conv1.forward(ac3);
    Tensor<Data> ac5 = ReLu<Data>(ac4);
    Tensor<Data> ac6 = conv2.forward(ac5);
    Tensor<Data> ac7 = ReLu<Data>(ac6 + ac3);

    std::cout << "----- Block 2-2 -----" << std::endl;

    // Block 2-2
    Tensor<Data> ac8 = conv3.forward(ac7);
    Tensor<Data> ac9 = ReLu<Data>(ac8);
    Tensor<Data> ac10 = conv4.forward(ac9);
    Tensor<Data> ac11 = ReLu<Data>(ac10 + ac7);

    std::cout << "----- Block 3-1 -----" << std::endl;

    // Block 3-1
    Tensor<Data> ac12 = conv5.forward(ac11);
    Tensor<Data> ac13 = ReLu<Data>(ac12);
    Tensor<Data> ac14 = conv6.forward(ac13);
    // short cut
    Tensor<Data> ac15 =
        ReLu<Data>(ac14 + conv7.forward(ac11));

    std::cout << "----- Block 3-2 -----" << std::endl;

    // Block 3-2
    Tensor<Data> ac16 = conv8.forward(ac15);
    Tensor<Data> ac17 = ReLu<Data>(ac16);
    Tensor<Data> ac18 = conv9.forward(ac17);
    Tensor<Data> ac19 = ReLu<Data>(ac18 + ac15);
    std::cout << "----- Block 4-1 -----" << std::endl;

    // Block 4-1
    Tensor<Data> ac20 = conv10.forward(ac19);
    Tensor<Data> ac21 = ReLu<Data>(ac20);
    Tensor<Data> ac22 = conv11.forward(ac21);
    // short cut
    Tensor<Data> ac23 =
        ReLu<Data>(ac22 + conv12.forward(ac19));
    std::cout << "----- Block 4-2 -----" << std::endl;

    // Block 4-2
    Tensor<Data> ac24 = conv13.forward(ac23);
    Tensor<Data> ac25 = ReLu<Data>(ac24);
    Tensor<Data> ac26 = conv14.forward(ac25);
    Tensor<Data> ac27 = ReLu<Data>(ac26 + ac23);
    std::cout << "----- Block 5-1 -----" << std::endl;

    // Block 5-1
    Tensor<Data> ac28 = conv15.forward(ac27);
    Tensor<Data> ac29 = ReLu<Data>(ac28);
    Tensor<Data> ac30 = conv16.forward(ac29);
    // short cut
    Tensor<Data> ac31 =
        ReLu<Data>(ac30 + conv17.forward(ac27));
    std::cout << "----- Block 5-2 -----" << std::endl;

    // Block 5-2
    Tensor<Data> ac32 = conv18.forward(ac31);
    Tensor<Data> ac33 = ReLu<Data>(ac32);
    Tensor<Data> ac34 = conv19.forward(ac33);
    Tensor<Data> ac35 = ReLu<Data>(ac34 + ac31);
    return ac35;
  }
};
#endif  // MODULE_H