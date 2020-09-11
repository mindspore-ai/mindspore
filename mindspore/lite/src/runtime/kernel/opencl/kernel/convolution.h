/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONVOLUTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONVOLUTION_H_

#include <vector>
#include <string>
#include "src/tensor.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "schema/model_generated.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::kernel {

class ConvolutionOpenCLKernel : public OpenCLKernel {
 public:
  explicit ConvolutionOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}
  ~ConvolutionOpenCLKernel() override{};

  int Init() override;
  int Run() override;
  int InitBuffer();
  int GetImageSize(size_t idx, std::vector<size_t> *img_size) override;

 private:
  bool use_fp16_ = false;

  int CI_{};
  int IH_{};
  int IW_{};
  int CO_{};
  int OH_{};
  int OW_{};
  int CI_SLICES_{};
  int CO_SLICES_{};
  int KH_{};
  int KW_{};
  void *packed_weight_ = nullptr;
  void *packed_bias_ = nullptr;

  bool use_winograd_ = false;
  int TILES_X_{};
  int TILES_Y_{};
  int TILES_XY_{};
  void *winograd_mem0_ = nullptr;
  void *winograd_mem1_ = nullptr;

  cl::Kernel kernel_4x4to36_;
  cl::Kernel kernel_conv_;
  cl::Kernel kernel_36to4x4_;

  int InitWeight();
  int InitBias();
  int RearrangeWinogradWeight();
  template <typename SRC_T, typename DST_T>
  int OHWI2OHWIOGroupI4O4(void *weight_OHWI, size_t KH, size_t KW, size_t OGroup);

  std::string CodeGenConvolutionNHWC4();
  std::string CodeGenConvolutionNC4HW4();

  std::string CodeGenWinograd4x4To36();
  std::string CodeGenWinogradConvolution();
  std::string CodeGenWinograd36To4x4();
  int SetGlobalLocalConv(std::vector<size_t> *global, std::vector<size_t> *local);

  size_t sizeof_FLT() const { return use_fp16_ ? sizeof(float16_t) : sizeof(float); }

  bool UseWinograd4x4To6x6() {
    auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
    const bool attr_valid = param->kernel_h_ == 3 && param->kernel_w_ == 3 && param->dilation_h_ == 1 &&
                            param->dilation_w_ == 1 && param->stride_h_ == 1 && param->stride_w_ == 1;
    const bool channel_good = CI_SLICES_ >= 12 && CO_SLICES_ >= 12;
    const bool hw_good = TILES_X_ * TILES_Y_ >= 16;
    return attr_valid && channel_good && hw_good;
  }

  static std::vector<float> MatrixMultiply(const float A[], const float B[], int M, int N, int K) {
    std::vector<float> C(M * K);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        float s = 0.0f;
        for (int k = 0; k < N; ++k) {
          s += A[i * N + k] * B[k * K + j];
        }
        C[i * K + j] = s;
      }
    }
    return C;
  }

  static int GetBiggestDivider(int x, int y) {
    for (int i = y; i != 0; i--) {
      if (x % i == 0) {
        return i;
      }
    }
    return 1;
  }
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONVOLUTION_H_
