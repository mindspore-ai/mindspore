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

#include "src/litert/kernel/cpu/fp32/reduce_scatter_fp32.h"
#include "schema/ops_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_ReduceScatter;

namespace mindspore::kernel {
int ReduceScatterCPUKernel::Prepare() {
  MS_LOG(ERROR) << "unsupported ReduceScatter kernel";
  return lite::RET_NOT_SUPPORT;
}

int ReduceScatterCPUKernel::ReSize() { return lite::RET_OK; }

void ReduceScatterCPUKernel::InitReduceData(void *data, size_t data_num) {
  if (data_type_ == kNumberTypeFloat32) {
    float *float_d = reinterpret_cast<float *>(data);
    if (param_->mode_ == schema::ReduceMode_ReduceProd) {
      for (size_t i = 0; i < data_num; i++) float_d[i] = 1.0f;
    } else if (param_->mode_ == schema::ReduceMode_ReduceMax) {
      for (size_t i = 0; i < data_num; i++) float_d[i] = FLT_MIN;
    } else if (param_->mode_ == schema::ReduceMode_ReduceMin) {
      for (size_t i = 0; i < data_num; i++) float_d[i] = FLT_MAX;
    } else {
      memset(data, 0, data_num * sizeof(float));
    }
  }
  return;
}

int ReduceScatterCPUKernel::DoReduceScatter(void *in_data, void *reduce_data, size_t data_size) {
  if (data_type_ == kNumberTypeFloat32) {
    float *in = reinterpret_cast<float *>(in_data);
    float *out = reinterpret_cast<float *>(reduce_data);

    if (param_->mode_ == schema::ReduceMode_ReduceSum) {
      for (size_t i = 0; i < data_size; i++) out[i] += in[i];
    } else if (param_->mode_ == schema::ReduceMode_ReduceMean) {
      for (size_t i = 0; i < data_size; i++) out[i] += (in[i] / static_cast<float>(param_->rank_size_));
    } else if (param_->mode_ == schema::ReduceMode_ReduceMax) {
      for (size_t i = 0; i < data_size; i++) out[i] = in[i] > out[i] ? in[i] : out[i];
    } else if (param_->mode_ == schema::ReduceMode_ReduceMin) {
      for (size_t i = 0; i < data_size; i++) out[i] = in[i] < out[i] ? in[i] : out[i];
    } else {
      MS_LOG(ERROR) << "unsupported mode in reduce scatter : " << param_->mode_;
      return lite::RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "unsupported data type in reduce scatter : " << data_type_;
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int ReduceScatterCPUKernel::Run() {
  int rank = param_->rank_size_;
  size_t in_data_size = in_tensors().front()->Size();
  size_t in_ele_size = in_tensors().front()->ElementsNum();
  size_t out_data_size = out_tensors().front()->Size();
  data_type_ = in_tensors().front()->data_type();

  void *reduce_data = ms_context_->allocator->Malloc(in_data_size);
  InitReduceData(reduce_data, in_ele_size);

  for (int i = 0; i < rank; i++) {
    /* update in_tensor by rank id */
    auto in_tensor = in_tensors().front();
    DoReduceScatter(in_tensor->data(), reduce_data, in_ele_size);
  }

  for (int i = 0; i < rank; i++) {
    /* update out_tensor by rank id */
    auto out_tensor = out_tensors().front();
    memcpy(out_tensor->data(), (reinterpret_cast<uint8_t *>(reduce_data)) + i * out_data_size, out_data_size);
  }

  ms_context_->allocator->Free(reduce_data);
  return lite::RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ReduceScatter, LiteKernelCreator<ReduceScatterCPUKernel>)
}  // namespace mindspore::kernel
