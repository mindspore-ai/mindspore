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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_NNACL_NNACL_LIB_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_NNACL_NNACL_LIB_H_

#include <vector>
#include "src/extendrt/kernel/kernel_lib.h"
#include "src/litert/kernel/cpu/nnacl/nnacl_kernel.h"
#include "src/common/tensor_util.h"

using mindspore::infer::abstract::Tensor;

namespace mindspore {
namespace kernel {
constexpr char kNNAclName[] = "NNAcl";
class NNAclLib : public KernelLib {
 public:
  NNAclLib() : KernelLib(kNNAclName, "CPU") {}

  bool Support(const PrimitiveType &op_type, const KernelAttr &dt, const Format &format) const override;
  LiteKernel *CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                           const std::vector<InferTensor *> &outputs, const InferContext *ctx) const override;
};
}  // namespace kernel
}  // namespace mindspore
#endif
