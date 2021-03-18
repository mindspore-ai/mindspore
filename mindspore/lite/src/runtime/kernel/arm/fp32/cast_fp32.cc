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
#include "src/runtime/kernel/arm/fp32/cast_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {
namespace {
int CastRun(void *cdata, int task_id) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }

  return reinterpret_cast<CastCPUKernel *>(cdata)->DoCast(task_id);
}
}  // namespace

int CastCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CastCPUKernel::ReSize() {
  data_num_ = in_tensors_.at(0)->ElementsNum();
  if (data_num_ == 0) {
    return RET_OK;
  }
  op_parameter_->thread_num_ = MSMIN(op_parameter_->thread_num_, static_cast<int>(data_num_));
  stride_ = UP_DIV(data_num_, op_parameter_->thread_num_);
  return RET_OK;
}

int CastCPUKernel::CastToFp32(lite::Tensor *input, lite::Tensor *output, int offset, int data_num) {
  auto input_data_type = input->data_type();
  auto output_data = output->data_c();
  switch (input_data_type) {
    case kNumberTypeBool:
      BoolToFloat32(reinterpret_cast<bool *>(input->MutableData()) + offset,
                    reinterpret_cast<float *>(output_data) + offset, data_num);
      break;
    case kNumberTypeUInt8:
      Uint8ToFloat32(reinterpret_cast<uint8_t *>(input->MutableData()) + offset,
                     reinterpret_cast<float *>(output_data) + offset, data_num);
      break;
    case kNumberTypeInt32:
      Int32ToFloat32(reinterpret_cast<int32_t *>(input->MutableData()) + offset,
                     reinterpret_cast<float *>(output_data) + offset, data_num);
      break;
    case kNumberTypeFloat16:
      Fp16ToFloat32(reinterpret_cast<uint16_t *>(input->MutableData()) + offset,
                    reinterpret_cast<float *>(output_data) + offset, data_num);
      break;
    case kNumberTypeInt64:
      Int64ToFloat32(reinterpret_cast<int64_t *>(input->MutableData()) + offset,
                     reinterpret_cast<float *>(output_data) + offset, data_num);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported input data type " << input_data_type;
      return RET_ERROR;
  }
  return RET_OK;
}

int CastCPUKernel::DoCast(int thread_id) {
  auto input = in_tensors_.at(0);
  int data_num = MSMIN(stride_, data_num_ - thread_id * stride_);
  if (data_num <= 0) {
    return RET_OK;
  }

  auto offset = thread_id * stride_;
  auto output = out_tensors_.at(0);
  auto output_data = output->data_c();
  MS_ASSERT(output_data != nullptr);
  auto input_data_type = input->data_type();
  auto output_data_type = output->data_type();
  if (input_data_type == output_data_type) {
    auto datalen = lite::DataTypeSize(input_data_type);
    memcpy(reinterpret_cast<char *>(output_data) + offset * datalen,
           reinterpret_cast<char *>(input->data_c()) + offset * datalen, data_num * datalen);
    return RET_OK;
  }
  if (output_data_type != kNumberTypeFloat32) {
    if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt64) {
      Float32ToInt64(reinterpret_cast<float *>(input->data_c()) + offset,
                     reinterpret_cast<int64_t *>(output_data) + offset, data_num);
    } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt32) {
      Float32ToInt32(reinterpret_cast<float *>(input->data_c()) + offset,
                     reinterpret_cast<int32_t *>(output_data) + offset, data_num);
    } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeFloat16) {
      Float32ToFp16(reinterpret_cast<float *>(input->data_c()) + offset,
                    reinterpret_cast<uint16_t *>(output_data) + offset, data_num);
    } else if (input_data_type == kNumberTypeInt32 && output_data_type == kNumberTypeInt64) {
      Int32ToInt64(reinterpret_cast<int32_t *>(input->data_c()) + offset,
                   reinterpret_cast<int64_t *>(output_data) + offset, data_num);
    } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt16) {
      Float32ToInt16(reinterpret_cast<float *>(input->data_c()) + offset,
                     reinterpret_cast<int16_t *>(output_data) + offset, data_num);
    } else if (input_data_type == kNumberTypeBool && output_data_type == kNumberTypeInt32) {
      BoolToInt32(reinterpret_cast<bool *>(input->data_c()) + offset, reinterpret_cast<int32_t *>(output_data) + offset,
                  data_num);
#ifdef ENABLE_FP16
    } else if (input_data_type == kNumberTypeInt64 && output_data_type == kNumberTypeFloat16) {
      Int64ToFp16(reinterpret_cast<int64_t *>(input->data_c()) + offset,
                  reinterpret_cast<float16_t *>(output_data) + offset, data_num);
#endif
    } else {
      MS_LOG(ERROR) << "Unsupported datatype from " << input_data_type << " to " << output_data_type;
      return RET_ERROR;
    }
  } else {
    return CastToFp32(input, output, offset, data_num);
  }
  return RET_OK;
}

int CastCPUKernel::Run() {
  if (data_num_ == 0) {
    return RET_OK;
  }
  return ParallelLaunch(this->context_->thread_pool_, CastRun, this, op_parameter_->thread_num_);
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Cast, LiteKernelCreator<CastCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt8, PrimitiveType_Cast, LiteKernelCreator<CastCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Cast, LiteKernelCreator<CastCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Cast, LiteKernelCreator<CastCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimitiveType_Cast, LiteKernelCreator<CastCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Cast, LiteKernelCreator<CastCPUKernel>)
#ifndef ENABLE_ARM
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Cast, LiteKernelCreator<CastCPUKernel>)
#endif
}  // namespace mindspore::kernel
