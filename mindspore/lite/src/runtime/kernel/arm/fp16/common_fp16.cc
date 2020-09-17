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

#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/cast_fp16.h"

namespace mindspore::kernel {
float16_t *ConvertInputFp32toFp16(lite::Tensor *input, const lite::InnerContext *ctx) {
  float16_t *fp16_data = nullptr;
  auto data_type = input->data_type();
  if (data_type == kNumberTypeFloat32) {
    auto ele_num = input->ElementsNum();
    fp16_data = reinterpret_cast<float16_t *>(ctx->allocator->Malloc(ele_num * sizeof(float16_t)));
    if (fp16_data == nullptr) {
      MS_LOG(ERROR) << "malloc fp16_data failed.";
      return nullptr;
    }
    auto ori_data = reinterpret_cast<float *>(input->MutableData());
    Float32ToFloat16(ori_data, fp16_data, ele_num);
  } else {
    fp16_data = reinterpret_cast<float16_t *>(input->MutableData());
  }
  return fp16_data;
}

float16_t *MallocOutputFp16(lite::Tensor *output, const lite::InnerContext *ctx) {
  float16_t *fp16_data = nullptr;
  auto data_type = output->data_type();
  if (data_type == kNumberTypeFloat32) {
    auto ele_num = output->ElementsNum();
    fp16_data = reinterpret_cast<float16_t *>(ctx->allocator->Malloc(ele_num * sizeof(float16_t)));
    if (fp16_data == nullptr) {
      MS_LOG(ERROR) << "malloc fp16_data failed.";
      return nullptr;
    }
  } else {
    fp16_data = reinterpret_cast<float16_t *>(output->MutableData());
  }
  return fp16_data;
}

bool IsExistFp16Tensor(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs) {
  bool result = false;
  for (auto &input : inputs) {
    if (input->data_type() == kNumberTypeFloat16) {
      result = true;
      break;
    }
  }
  if (result) {
    return true;
  }
  for (auto &output : outputs) {
    if (output->data_type() == kNumberTypeFloat16) {
      result = true;
      break;
    }
  }
  return result;
}
}  // namespace mindspore::kernel
