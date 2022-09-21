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
#include "src/litert/kernel/cpu/string/hashtable_lookup.h"
#include <string>
#include <algorithm>
#include "src/litert/kernel_registry.h"
#include "src/common/string_utils.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_HashtableLookup;

namespace mindspore::kernel {
int HashtableLookupCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C3NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C2NUM);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int HashtableLookupCPUKernel::ReSize() { return RET_OK; }

static int CmpKeyFunc(const void *lhs, const void *rhs) {
  return *static_cast<const int *>(lhs) - *static_cast<const int *>(rhs);
}

int HashtableLookupCPUKernel::Run() {
  auto input_tensor = in_tensors_[FIRST_INPUT];
  auto keys_tensor = in_tensors_[SECOND_INPUT];
  auto values_tensor = in_tensors_[THIRD_INPUT];
  auto output_tensor = out_tensors_[FIRST_INPUT];
  auto hits_tensor = out_tensors_[SECOND_INPUT];
  CHECK_NULL_RETURN(input_tensor);
  CHECK_NULL_RETURN(keys_tensor);
  CHECK_NULL_RETURN(values_tensor);
  CHECK_NULL_RETURN(output_tensor);
  CHECK_NULL_RETURN(hits_tensor);

  int rows = GetStringCount(values_tensor);
  if (rows < 0) {
    MS_LOG(ERROR) << "get string cnt fail!";
    return RET_ERROR;
  }
  int32_t *input_data = reinterpret_cast<int32_t *>(input_tensor->MutableData());
  uint8_t *hits_data = reinterpret_cast<uint8_t *>(hits_tensor->MutableData());
  std::vector<lite::StringPack> output_string_pack(input_tensor->ElementsNum());
  std::vector<lite::StringPack> all_string_pack = ParseTensorBuffer(values_tensor);
  lite::StringPack null_string_pack = {0, nullptr};

  for (int i = 0; i < input_tensor->ElementsNum(); i++) {
    int index = -1;
    void *p = bsearch(&(input_data[i]), keys_tensor->MutableData(), rows, sizeof(int32_t), CmpKeyFunc);
    if (p != nullptr) {
      index = reinterpret_cast<int32_t *>(p) - reinterpret_cast<int32_t *>(keys_tensor->MutableData());
    }
    if (index >= rows || index < 0) {
      output_string_pack[i] = null_string_pack;
      hits_data[i] = 0;
    } else {
      output_string_pack[i] = all_string_pack[index];
      hits_data[i] = 1;
    }
  }
  WriteStringsToTensor(output_tensor, output_string_pack);
  output_tensor->ResetRefCount();  // ref-count of data will be lost due to "WriteStringsToTensor"
  return RET_OK;
}

kernel::LiteKernel *CpuHashtableLookupKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                    const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                                    const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  auto *kernel = new (std::nothrow) HashtableLookupCPUKernel(parameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new HashtableLookupCPUKernel fail!";
    free(parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_HashtableLookup, CpuHashtableLookupKernelCreator)
}  // namespace mindspore::kernel
