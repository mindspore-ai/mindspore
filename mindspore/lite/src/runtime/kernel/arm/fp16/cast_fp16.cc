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
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {
namespace {
int CastFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }

  return reinterpret_cast<CastFp16CPUKernel *>(cdata)->DoCast(task_id);
}
}  // namespace

int CastFp16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CastFp16CPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  auto in_tensor = in_tensors_.at(0);
  CHECK_NULL_RETURN(in_tensor);
  data_num_ = in_tensor->ElementsNum();
  if (data_num_ == 0) {
    return RET_OK;
  }
  op_parameter_->thread_num_ = MSMIN(op_parameter_->thread_num_, data_num_);
  stride_ = UP_DIV(data_num_, op_parameter_->thread_num_);
  return RET_OK;
}

int CastFp16CPUKernel::DoCast(int thread_id) {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(output);
  auto input_data = input->data();
  auto output_data = output->data();
  CHECK_NULL_RETURN(input_data);
  CHECK_NULL_RETURN(output_data);

  int data_num = MSMIN(stride_, data_num_ - thread_id * stride_);
  if (data_num <= 0) {
    return RET_OK;
  }

  auto offset = thread_id * stride_;
  auto input_data_type = input->data_type();
  auto output_data_type = output->data_type();

  if (input_data_type == kNumberTypeFloat16) {
    switch (output_data_type) {
      case kNumberTypeInt64:
        Float16ToInt64(reinterpret_cast<float16_t *>(input_data) + offset,
                       reinterpret_cast<int64_t *>(output_data) + offset, data_num);
        break;
      case kNumberTypeInt32:
        Float16ToInt32(reinterpret_cast<float16_t *>(input_data) + offset,
                       reinterpret_cast<int32_t *>(output_data) + offset, data_num);
        break;
      case kNumberTypeFloat32:
        Float16ToFloat32(reinterpret_cast<float16_t *>(input_data) + offset,
                         reinterpret_cast<float *>(output_data) + offset, data_num);
        break;
      case kNumberTypeFloat16:
        memcpy(reinterpret_cast<float16_t *>(output_data) + offset, reinterpret_cast<float16_t *>(input_data) + offset,
               data_num * sizeof(float16_t));
        break;
      default:
        MS_LOG(ERROR) << "Unsupported output data type " << output_data_type;
        return RET_ERROR;
    }
  } else if (input_data_type == kNumberTypeFloat32) {
    switch (output_data_type) {
      case kNumberTypeInt64:
        Float32ToInt64(reinterpret_cast<float *>(input_data) + offset,
                       reinterpret_cast<int64_t *>(output_data) + offset, data_num);
        break;
      case kNumberTypeInt32:
        Float32ToInt32(reinterpret_cast<float *>(input_data) + offset,
                       reinterpret_cast<int32_t *>(output_data) + offset, data_num);
        break;
      case kNumberTypeFloat32:
        memcpy(reinterpret_cast<float *>(output_data) + offset, reinterpret_cast<float *>(input_data) + offset,
               data_num * sizeof(float));
        break;
      case kNumberTypeFloat16:
        Float32ToFloat16(reinterpret_cast<float *>(input_data) + offset,
                         reinterpret_cast<float16_t *>(output_data) + offset, data_num);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported output data type " << output_data_type;
        return RET_ERROR;
    }
  } else if (input_data_type == kNumberTypeInt32) {
    switch (output_data_type) {
      case kNumberTypeFloat32:
        Int32ToFloat32(static_cast<int32_t *>(input_data) + offset, static_cast<float *>(output_data) + offset,
                       data_num);
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
  return ParallelLaunch(this->ms_context_, CastFp16Run, this, op_parameter_->thread_num_);
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Cast, LiteKernelCreator<CastFp16CPUKernel>)
}  // namespace mindspore::kernel
