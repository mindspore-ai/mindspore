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
#include "src/runtime/kernel/arm/fp16/cast_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {
namespace {
int CastFp16Run(void *cdata, int task_id) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }

  return reinterpret_cast<CastFp16CPUKernel *>(cdata)->DoCast(task_id);
}
}  // namespace

int CastFp16CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CastFp16CPUKernel::ReSize() {
  data_num_ = in_tensors_.at(0)->ElementsNum();
  if (data_num_ == 0) {
    return RET_OK;
  }
  op_parameter_->thread_num_ = MSMIN(op_parameter_->thread_num_, static_cast<int>(data_num_));
  stride_ = UP_DIV(data_num_, op_parameter_->thread_num_);
  return RET_OK;
}

int CastFp16CPUKernel::DoCast(int thread_id) {
  auto input = in_tensors_.at(0);
  int data_num = MSMIN(stride_, data_num_ - thread_id * stride_);
  if (data_num <= 0) {
    return RET_OK;
  }

  auto offset = thread_id * stride_;
  auto output = out_tensors_.at(0);
  auto output_data = output->data_c();
  auto input_data_type = input->data_type();
  auto output_data_type = output->data_type();

  if (input_data_type == kNumberTypeFloat16) {
    switch (output_data_type) {
      case kNumberTypeInt64:
        Float16ToInt64(reinterpret_cast<float16_t *>(input->data_c()) + offset,
                       reinterpret_cast<int64_t *>(output_data) + offset, data_num);
        break;
      case kNumberTypeInt32:
        Float16ToInt32(reinterpret_cast<float16_t *>(input->data_c()) + offset,
                       reinterpret_cast<int32_t *>(output_data) + offset, data_num);
        break;
      case kNumberTypeFloat32:
        Float16ToFloat32(reinterpret_cast<float16_t *>(input->MutableData()) + offset,
                         reinterpret_cast<float *>(output_data) + offset, data_num);
        break;
      case kNumberTypeFloat16:
        memcpy(reinterpret_cast<float16_t *>(output_data) + offset,
               reinterpret_cast<float16_t *>(input->data_c()) + offset, data_num * sizeof(float16_t));
        break;
      default:
        MS_LOG(ERROR) << "Unsupported output data type " << output_data_type;
        return RET_ERROR;
    }
  } else if (input_data_type == kNumberTypeFloat32) {
    switch (output_data_type) {
      case kNumberTypeInt64:
        Float32ToInt64(reinterpret_cast<float *>(input->data_c()) + offset,
                       reinterpret_cast<int64_t *>(output_data) + offset, data_num);
        break;
      case kNumberTypeInt32:
        Float32ToInt32(reinterpret_cast<float *>(input->data_c()) + offset,
                       reinterpret_cast<int32_t *>(output_data) + offset, data_num);
        break;
      case kNumberTypeFloat32:
        memcpy(reinterpret_cast<float *>(output_data) + offset, reinterpret_cast<float *>(input->data_c()) + offset,
               data_num * sizeof(float));
        break;
      case kNumberTypeFloat16:
        Float32ToFloat16(reinterpret_cast<float *>(input->MutableData()) + offset,
                         reinterpret_cast<float16_t *>(output_data) + offset, data_num);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported output data type " << output_data_type;
        return RET_ERROR;
    }
  } else if (input_data_type == kNumberTypeInt32) {
    switch (output_data_type) {
      case kNumberTypeFloat32:
        Int32ToFloat32(static_cast<int32_t *>(input->data_c()) + offset, static_cast<float *>(output_data) + offset,
                       data_num);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported output data type " << output_data_type;
        return RET_ERROR;
    }
  } else if (input_data_type == kNumberTypeInt64) {
    switch (output_data_type) {
      case kNumberTypeFloat16:
        Int64ToFloat32(reinterpret_cast<int64_t *>(input->MutableData()) + offset,
                       reinterpret_cast<float *>(output_data) + offset, data_num);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported output data type " << output_data_type;
        return RET_ERROR;
    }

  } else {
    MS_LOG(ERROR) << "Unsupported input data type " << input_data_type;
    return RET_ERROR;
  }
  return RET_OK;
}

int CastFp16CPUKernel::Run() {
  if (data_num_ == 0) {
    return RET_OK;
  }
  return ParallelLaunch(this->context_->thread_pool_, CastFp16Run, this, op_parameter_->thread_num_);
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Cast, LiteKernelCreator<CastFp16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimitiveType_Cast, LiteKernelCreator<CastFp16CPUKernel>)
}  // namespace mindspore::kernel
