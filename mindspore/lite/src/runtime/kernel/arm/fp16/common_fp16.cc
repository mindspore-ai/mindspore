/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

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

int ConvertFp32TensorToFp16(lite::Tensor *tensor, const lite::InnerContext *ctx) {
  if (tensor->data_type() == TypeId::kNumberTypeFloat16) {
    return RET_OK;
  }
  auto fp32_data = tensor->data_c();
  tensor->set_data(nullptr);
  tensor->set_data_type(TypeId::kNumberTypeFloat16);
  auto ret = tensor->MallocData();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_ERROR;
  }
  Float32ToFloat16(static_cast<float *>(fp32_data), static_cast<float16_t *>(tensor->data_c()), tensor->ElementsNum());
  ctx->allocator->Free(fp32_data);
  return RET_OK;
}
}  // namespace mindspore::kernel
