/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_VSL_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_VSL_KERNEL_H_

#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include "extendrt/delegate/ascend_native/ascend_native_base_kernel.h"
#include "extendrt/utils/func_graph_utils.h"

namespace mindspore::kernel {
class AscendNativeVslKernel : public AscendNativeBaseKernel {
 public:
  AscendNativeVslKernel(const std::vector<InferTensor *> &inputs, const std::vector<InferTensor *> &outputs,
                        InferPrimitive prim, const InferContext *ctx, const void *stream, std::string name,
                        const void *acl_ctx_)
      : AscendNativeBaseKernel(inputs, outputs, prim, ctx, stream, name, acl_ctx_) {}
  virtual ~AscendNativeVslKernel() {}

  int Prepare() override;

  int Run() override;

  int InferShape() override;

  int ReSize() override;

 private:
  int batch_size_{0};
  int seq_{0};
  int32_t *tmp0;
  int32_t *tmp1;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_VSL_KERNEL_H_
