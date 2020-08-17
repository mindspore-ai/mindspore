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
#include "src/ir/tensor.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "schema/model_generated.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/arm/nnacl/conv_parameter.h"

namespace mindspore::kernel {

class ConvolutionOpenCLKernel : public OpenCLKernel {
 public:
  explicit ConvolutionOpenCLKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                                   const std::vector<lite::tensor::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}
  ~ConvolutionOpenCLKernel() override{};

  int Init() override;
  int Run() override;
  int InitBuffer();
  int GetImageSize(size_t idx, std::vector<size_t> *img_size) override;

 private:
  int CI_SLICES;
  int CO_SLICES;
  float *packed_weight_ = nullptr;
  float *packed_bias_ = nullptr;

  bool use_winograd_ = false;
  int TILES_X;
  int TILES_Y;
  int TILES_XY;
  void *winograd_mem0_ = nullptr;
  void *winograd_mem1_ = nullptr;

  cl::Kernel kernel_4x4to36;
  cl::Kernel kernel_conv;
  cl::Kernel kernel_36to4x4;

  std::string CodeGenConvolution();
  std::string CodeGenWinograd4x4To36();
  std::string CodeGenWinogradConvolution();
  std::string CodeGenWinograd36To4x4();
  int SetGlobalLocalConv(std::vector<size_t> *global, std::vector<size_t> *local);

  bool UseWinograd4x4To6x6() {
    auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
    const bool attr_valid = param->kernel_h_ == 3 && param->kernel_w_ == 3 && param->dilation_h_ == 1 &&
                            param->dilation_w_ == 1 && param->stride_h_ == 1 && param->stride_w_ == 1;
    const bool channel_good = CO_SLICES % 4 == 0 && CI_SLICES >= 16 && CO_SLICES >= 16;
    const bool hw_good = TILES_X * TILES_Y >= 32;
    return attr_valid && channel_good && hw_good;
  }

  std::vector<float> MatrixMultiply(const std::vector<float> &A, const std::vector<float> &B, int M, int N, int K) {
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
