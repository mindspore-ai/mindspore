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
#include "nnacl/conv_parameter.h"
#include "schema/ops_generated.h"

namespace mindspore::kernel {

class ConvolutionOpenCLKernel : public OpenCLKernel {
 public:
  ConvolutionOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}
  ~ConvolutionOpenCLKernel() override = default;

  int Init() override;
  int Run() override;
  int InitBuffer() override;

 private:
  int InitWeight();
  int InitBias();
  int GenerateWinogradWeight();
  std::string CodeGenConvolutionNHWC4();
  std::string CodeGenConvolutionNC4HW4();
  std::string CodeGenWinograd4x4To36();
  std::string CodeGenWinogradConvolution();
  std::string CodeGenWinograd36To4x4();
  int SetGlobalLocalConv(std::vector<size_t> *global, std::vector<size_t> *local);

  size_t sizeof_FLT() const { return use_fp16_ ? sizeof(float16_t) : sizeof(float); }

  bool UseWinograd4x4To6x6() {
    auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
    const bool attr_valid = param->kernel_h_ == 3 && param->kernel_w_ == 3 && param->stride_h_ == 1 &&
                            param->stride_w_ == 1 && param->pad_u_ == 1 && param->pad_d_ == 1 && param->pad_l_ == 1 &&
                            param->pad_r_ == 1 && param->dilation_h_ == 1 && param->dilation_w_ == 1 && IH_ == OH_ &&
                            IW_ == OW_ && batch_size_ == 1;
    const bool channel_good = CI_SLICES_ >= 12 && CO_SLICES_ >= 12;
    const bool hw_good = TILES_X_ * TILES_Y_ >= 16;
    return attr_valid && channel_good && hw_good;
  }

  std::string get_code_id() {
    auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
    std::vector<int> vpara{batch_size_,
                           CI_,
                           IH_,
                           IW_,
                           CO_,
                           OH_,
                           OW_,
                           KH_,
                           KW_,
                           param->stride_h_,
                           param->stride_w_,
                           param->pad_u_,
                           param->pad_l_,
                           param->pad_d_,
                           param->pad_r_,
                           param->dilation_h_,
                           param->dilation_w_,
                           has_bias_,
                           use_fp16_,
                           op_format_,
                           param->act_type_};
    std::string code_id;
    for (auto &iv : vpara) {
      code_id += "_" + std::to_string(iv);
    }
    return code_id;
  }

  bool use_fp16_{false};
  const schema::Format op_format_{schema::Format_NHWC4};

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

  cl::Kernel kernel_4x4to36_;
  cl::Kernel kernel_conv_;
  cl::Kernel kernel_36to4x4_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONVOLUTION_H_
