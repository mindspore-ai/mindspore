/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CONVOLUTION_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CONVOLUTION_BASE_H_

#include <vector>
#include <unordered_map>
#include <string>
#include <limits>
#ifdef ENABLE_ARM
#include <arm_neon.h>
#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#endif
#endif
#include "src/inner_kernel.h"
#include "include/context.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "src/weight_decoder.h"
#include "include/errorcode.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class ConvolutionBaseCPUKernel : public InnerKernel {
 public:
  ConvolutionBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx, void *origin_weight,
                           void *origin_bias)
      : InnerKernel(parameter, inputs, outputs, ctx),
        ctx_(ctx),
        thread_count_(op_parameter_->thread_num_),
        origin_weight_(origin_weight),
        origin_bias_(origin_bias) {
    conv_param_ = reinterpret_cast<ConvParameter *>(op_parameter_);
  }
  ~ConvolutionBaseCPUKernel() override;

  int Init() override;
  int ReSize() override { return 0; }
  int Run() override { return 0; }
  int SetIfPerChannel();
  int MallocQuantParam();
  int SetQuantParam();
  int SetInputTensorQuantParam();
  int SetFilterTensorQuantParam();
  int SetOutputTensorQuantParam();
  int SetQuantMultiplier();
  void SetRoundingAndMultipilerMode();
  int CheckResizeValid();
  void FreeQuantParam();
  void *MallocAlignedData(size_t alignment, size_t size);
  void FreeAlignedData(void **ptr);

 protected:
  int InitConvWeightBias();
  int RepackWeight();
  void UpdateOriginWeightAndBias();

  virtual int MallocWeightBiasData() { return RET_OK; }
  virtual void PackWeight() {}
  bool IsRepack() { return is_repack_; }
  std::unordered_map<uintptr_t, void *> addr_map;
  void *packed_weight_ = nullptr;
  void *bias_data_ = nullptr;
  const InnerContext *ctx_ = nullptr;
  ConvParameter *conv_param_ = nullptr;
  ConvQuantArg *conv_quant_arg_ = nullptr;
  int tile_num_ = 0;
  int thread_count_ = 1;
  bool is_repack_ = false;
  void *origin_weight_;  // do not free
  void *origin_bias_;    // do not free
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CONVOLUTION_BASE_H_
