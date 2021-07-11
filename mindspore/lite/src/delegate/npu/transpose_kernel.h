/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_TRANSPOSE_KERNEL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_TRANSPOSE_KERNEL_H_
#include <vector>
#include <string>
#include "include/graph/op/all_ops.h"
#include "include/api/kernel.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
void PackNHWCToNCHWFp32(const void *src, void *dst, int batches, int plane, int channel);

void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel);

class TransposeNPUKernel : public kernel::Kernel {
 public:
  TransposeNPUKernel(const std::vector<mindspore::MSTensor> &in_tensors,
                     const std::vector<mindspore::MSTensor> &out_tensors, std::vector<int> perm, std::string name)
      : kernel::Kernel(in_tensors, out_tensors, nullptr, nullptr) {
    type_ = schema::PrimitiveType_Transpose;
    name_ = name;
    perm_ = perm;
  }

  ~TransposeNPUKernel() override = default;

  int Prepare() override { return RET_OK; }

  int Execute() override;

  int ReSize() override {
    MS_LOG(ERROR) << "NPU does not support the resize function temporarily.";
    return lite::RET_ERROR;
  }

 protected:
  std::vector<int> perm_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_TRANSPOSE_KERNEL_H_
