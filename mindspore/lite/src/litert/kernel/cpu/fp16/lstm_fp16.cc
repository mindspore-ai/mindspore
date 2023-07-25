/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include <vector>
#include "src/litert/kernel/cpu/fp16/lstm_mindir_fp16.h"
#include "src/litert/kernel/cpu/fp16/lstm_non_mindir_fp16.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LSTM;

namespace mindspore::kernel {
namespace {
constexpr size_t kMindirInputTensorNum = 4;
}  // namespace

LiteKernel *LstmFp16KernelCreator(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  OpParameter *parameter, const lite::InnerContext *ctx,
                                  const kernel::KernelKey &desc) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr.";
    return nullptr;
  }
  if (desc.data_type == kTypeUnknown) {
    MS_LOG(WARNING) << "desc data_type is unknown.";
  }
  LiteKernel *kernel{nullptr};
  if (inputs.size() == kMindirInputTensorNum) {
    kernel = new (std::nothrow) LstmMindirFp16CPUKernel(parameter, inputs, outputs, ctx);
  } else {
    kernel = new (std::nothrow) LstmNonMindirFp16CPUKernel(parameter, inputs, outputs, ctx);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel: " << parameter->name_ << "is nullptr.";
    free(parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LSTM, LstmFp16KernelCreator)
}  // namespace mindspore::kernel
