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
#include "src/runtime/kernel/arm/fp16/quant_dtype_cast_fp16.h"
#include <vector>
#include "nnacl/int8/quant_dtype_cast_int8.h"
#include "nnacl/fp16/quant_dtype_cast_fp16.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_QuantDTypeCast;

namespace mindspore::kernel {
int QuantDTypeCastFp16CPUKernel::Init() {
  if (in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "inputs number should be 1, but " << in_tensors_.size() << " is given.";
    return RET_PARAM_INVALID;
  }
  if (out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "outputs number should be 1, but " << out_tensors_.size() << " is given.";
    return RET_PARAM_INVALID;
  }
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  auto param = reinterpret_cast<QuantDTypeCastParameter *>(op_parameter_);
  if (param->dstT == kNumberTypeInt8) {
    if (in_tensor->data_type() != kNumberTypeFloat16 || out_tensor->data_type() != kNumberTypeInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
    int_to_float_ = false;
    is_uint8_ = false;
  } else if (param->srcT == kNumberTypeInt8) {
    if (in_tensor->data_type() != kNumberTypeInt8 || out_tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
    int_to_float_ = true;
    is_uint8_ = false;
  } else if (param->dstT == kNumberTypeUInt8) {
    if (in_tensor->data_type() != kNumberTypeFloat16 || out_tensor->data_type() != kNumberTypeUInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
    int_to_float_ = false;
    is_uint8_ = true;
  } else if (param->srcT == kNumberTypeUInt8) {
    if (in_tensor->data_type() != kNumberTypeUInt8 || out_tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
    int_to_float_ = true;
    is_uint8_ = true;
  } else {
    MS_LOG(ERROR) << "param data type not supported:"
                  << " src: " << param->srcT << " dst: " << param->dstT;
    return RET_PARAM_INVALID;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int QuantDTypeCastFp16CPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  num_unit_ = static_cast<int>(in_tensor->ElementsNum());
  thread_n_num_ = MSMIN(thread_num_, num_unit_);
  thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  return RET_OK;
}

int QuantDTypeCastFp16CPUKernel::QuantDTypeCast(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  if (in_tensors_.front()->quant_params().empty() && out_tensors_.front()->quant_params().empty()) {
    MS_LOG(ERROR) << "QuantDTypeCast need quantization parameters which is not found.";
    return RET_ERROR;
  }
  auto quant_arg = !out_tensors_.front()->quant_params().empty() ? out_tensors_.front()->quant_params().front()
                                                                 : in_tensors_.front()->quant_params().front();
  int ret;
  MS_ASSERT(float16_ptr_);
  if (!is_uint8_) {
    MS_ASSERT(int8_ptr_);
    if (int_to_float_) {
      ret = DoDequantizeInt8ToFp16(int8_ptr_ + thread_offset, float16_ptr_ + thread_offset, quant_arg.scale,
                                   quant_arg.zeroPoint, num_unit_thread);
    } else {
      ret = DoQuantizeFp16ToInt8(float16_ptr_ + thread_offset, int8_ptr_ + thread_offset, quant_arg.scale,
                                 quant_arg.zeroPoint, num_unit_thread);
    }
  } else {
    // uint8
    MS_ASSERT(uint8_ptr_);
    if (int_to_float_) {
      ret = DoDequantizeUInt8ToFp16(uint8_ptr_ + thread_offset, float16_ptr_ + thread_offset, quant_arg.scale,
                                    quant_arg.zeroPoint, num_unit_thread);
    } else {
      ret = DoQuantizeFp16ToUInt8(float16_ptr_ + thread_offset, uint8_ptr_ + thread_offset, quant_arg.scale,
                                  quant_arg.zeroPoint, num_unit_thread);
    }
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCastFp16 error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantDTypeCastFP16Run(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<QuantDTypeCastFp16CPUKernel *>(cdata);
  auto ret = g_kernel->QuantDTypeCast(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCastFP16Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantDTypeCastFp16CPUKernel::Run() {
  if (in_tensors_.at(0)->data_type() == TypeId::kNumberTypeInt8 &&
      out_tensors_.at(0)->data_type() == TypeId::kNumberTypeFloat16) {
    int8_ptr_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data_c());
    float16_ptr_ = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data_c());
  } else if (in_tensors_.at(0)->data_type() == TypeId::kNumberTypeFloat16 &&
             out_tensors_.at(0)->data_type() == TypeId::kNumberTypeInt8) {
    float16_ptr_ = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data_c());
    int8_ptr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data_c());
  } else if (in_tensors_.at(0)->data_type() == TypeId::kNumberTypeUInt8 &&
             out_tensors_.at(0)->data_type() == TypeId::kNumberTypeFloat16) {
    uint8_ptr_ = reinterpret_cast<uint8_t *>(in_tensors_.at(0)->data_c());
    float16_ptr_ = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data_c());
  } else if (in_tensors_.at(0)->data_type() == TypeId::kNumberTypeFloat16 &&
             out_tensors_.at(0)->data_type() == TypeId::kNumberTypeUInt8) {
    float16_ptr_ = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data_c());
    uint8_ptr_ = reinterpret_cast<uint8_t *>(out_tensors_.at(0)->data_c());
  } else {
    MS_LOG(ERROR) << "QuantDTypeCastFp16 not support input or output type";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(this->context_->thread_pool_, QuantDTypeCastFP16Run, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuQuantDTypeCastFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                       const std::vector<lite::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::InnerContext *ctx,
                                                       const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) QuantDTypeCastFp16CPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new QuantDTypeCastFp16CPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed! name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_QuantDTypeCast, LiteKernelCreator<QuantDTypeCastFp16CPUKernel>)
}  // namespace mindspore::kernel
