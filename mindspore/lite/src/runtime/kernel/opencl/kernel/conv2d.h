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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONV2D_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONV2D_H_

#include <vector>
#include <string>
#include "src/tensor.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "schema/model_generated.h"
#include "nnacl/conv_parameter.h"
#include "schema/ops_generated.h"

namespace mindspore::kernel {

class Conv2DOpenCLKernel : public OpenCLKernel {
 public:
  Conv2DOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs), param_(reinterpret_cast<ConvParameter *>(parameter)) {}
  ~Conv2DOpenCLKernel() override = default;

  int CheckSpecs() override;

  int Prepare() override;
  void SetGlobalLocal() override;
  int InitWeights() override;
  void SetConstArgs() override;

  int Run() override;
  int Tune() override;
  std::vector<BaseTuningParameter> GenerateTuningParam() override;
  std::string Key() override;

  // for opencl fusion: Conv2D + PReLU(weight is scalar) -> param_.act_type=ActivationType_LEAKY_RELU
  float alpha_{0.0f};

 private:
  void SetBlockSize();
  int InitFilter();
  int InitBias();
  int GenerateWinogradFilter();

  bool UseWinograd4x4To6x6() {
    const bool attr_valid = param_->kernel_h_ == 3 && param_->kernel_w_ == 3 && param_->stride_h_ == 1 &&
                            param_->stride_w_ == 1 && param_->pad_u_ == 1 && param_->pad_d_ == 1 &&
                            param_->pad_l_ == 1 && param_->pad_r_ == 1 && param_->dilation_h_ == 1 &&
                            param_->dilation_w_ == 1 && IH_ == OH_ && IW_ == OW_ && batch_size_ == 1;
    const bool channel_good = CI_SLICES_ >= 8 && CO_SLICES_ >= 8;
    const bool hw_good = TILES_X_ * TILES_Y_ >= 16;
    return attr_valid && channel_good && hw_good;
  }

  cl::Kernel kernel_4x4to36_;
  cl::Kernel kernel_36to4x4_;
  cl::NDRange global_4x4to36_, local_4x4to36_;
  cl::NDRange global_conv_, local_conv_;
  cl::NDRange global_36to4x4_, local_36to4x4_;

  bool use_fp16_{false};
  size_t sizeof_FLT_{4};

  ConvParameter *param_{nullptr};
  int batch_size_{};
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
  void *packed_weight_{nullptr};
  void *packed_bias_{nullptr};
  bool has_bias_{false};

  bool use_winograd_{false};
  int TILES_X_{};
  int TILES_Y_{};
  int TILES_XY_{};
  void *winograd_mem0_{nullptr};
  void *winograd_mem1_{nullptr};

  struct {
    int H{1};
    int W{1};
    int C{1};
  } block_size_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONV2D_H_
