/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_ENCODER_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_ENCODER_KERNEL_H_

#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include "extendrt/delegate/ascend_native/ascend_native_base_kernel.h"
#include "extendrt/utils/func_graph_utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/encoder.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/query.h"

namespace mindspore::kernel {
class AscendNativeEncoderKernel : public AscendNativeBaseKernel {
 public:
  AscendNativeEncoderKernel(const std::vector<InferTensor *> &inputs, const std::vector<InferTensor *> &outputs,
                            InferPrimitive prim, const InferContext *ctx, const void *stream, std::string name)
      : AscendNativeBaseKernel(inputs, outputs, prim, ctx, stream, name),
        driver_input_tensors_(ENCODER_LAST_IDX),
        driver_output_tensors_(ENCODER_OUTPUT_LAST_IDX) {}
  virtual ~AscendNativeEncoderKernel() {}

  int Prepare() override;

  int Run() override;

  size_t get_workspace_size() const override;

  int InferShape() override;

 private:
  void build_driver_input_const_tensors();
  int InitEncoderParam();
  std::vector<int32_t> GetOutputDimensions();
  ascend_native::EncoderParams param_;
  std::shared_ptr<ascend_native::AscendNativeEncoder> encoder_driver_;
  std::vector<void *> driver_input_tensors_;
  std::vector<void *> driver_output_tensors_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_ENCODER_KERNEL_H_
