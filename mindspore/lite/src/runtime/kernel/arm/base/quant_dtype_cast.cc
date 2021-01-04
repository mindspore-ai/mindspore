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
#include "src/runtime/kernel/arm/base/quant_dtype_cast.h"
#include <vector>
#include "nnacl/int8/quant_dtype_cast_int8.h"
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
int QuantDTypeCastCPUKernel::Init() {
  if (in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "inputs number should be 1, but " << in_tensors_.size() << " is given.";
    return RET_PARAM_INVALID;
  }
  if (out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "outputs number should be 1, but " << out_tensors_.size() << " is given.";
    return RET_PARAM_INVALID;
  }
  auto in_tensor = in_tensors_.front();
  MS_ASSERT(in_tensor);
  auto out_tensor = out_tensors_.front();
  MS_ASSERT(out_tensor);
  auto param = reinterpret_cast<QuantDTypeCastParameter *>(op_parameter_);
  MS_ASSERT(param);
  if (param->srcT == kNumberTypeFloat32 && param->dstT == kNumberTypeInt8) {
    if (in_tensor->data_type() != kNumberTypeFloat32 || out_tensor->data_type() != kNumberTypeInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
  } else if (param->srcT == kNumberTypeInt8 && param->dstT == kNumberTypeFloat32) {
    if (in_tensor->data_type() != kNumberTypeInt8 || out_tensor->data_type() != kNumberTypeFloat32) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
  } else if (param->srcT == kNumberTypeUInt8 && param->dstT == kNumberTypeInt8) {
    if (in_tensor->data_type() != kNumberTypeUInt8 || out_tensor->data_type() != kNumberTypeInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
  } else if (param->srcT == kNumberTypeInt8 && param->dstT == kNumberTypeInt8) {
    if (in_tensor->data_type() != kNumberTypeInt8 || out_tensor->data_type() != kNumberTypeInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
  } else if (param->srcT == kNumberTypeInt8 && param->dstT == kNumberTypeUInt8) {
    if (in_tensor->data_type() != kNumberTypeInt8 || out_tensor->data_type() != kNumberTypeUInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
  } else if (param->srcT == kNumberTypeUInt8 && param->dstT == kNumberTypeFloat32) {
    if (in_tensor->data_type() != kNumberTypeUInt8 || out_tensor->data_type() != kNumberTypeFloat32) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
  } else if (param->srcT == kNumberTypeFloat32 && param->dstT == kNumberTypeUInt8) {
    if (in_tensor->data_type() != kNumberTypeFloat32 || out_tensor->data_type() != kNumberTypeUInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "param data type not supported:"
                  << " src: " << param->srcT << " dst: " << param->dstT;
    return RET_PARAM_INVALID;
  }
  src_dtype = param->srcT;
  dst_dtype = param->dstT;

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int QuantDTypeCastCPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  num_unit_ = static_cast<int>(in_tensor->ElementsNum());
  thread_n_num_ = MSMIN(thread_num_, num_unit_);
  thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  return RET_OK;
}

int QuantDTypeCastCPUKernel::QuantDTypeCast(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  if (in_tensors_.front()->quant_params().empty() && out_tensors_.front()->quant_params().empty()) {
    MS_LOG(ERROR) << "QuantDTypeCast need quantization parameters which is not found.";
    return RET_ERROR;
  }
  auto quant_arg =
    (!out_tensors_.front()->quant_params().empty() && out_tensors_.front()->quant_params().front().inited)
      ? out_tensors_.front()->quant_params().front()
      : in_tensors_.front()->quant_params().front();
  int ret = RET_OK;
  if (src_dtype == TypeId::kNumberTypeInt8 && dst_dtype == TypeId::kNumberTypeFloat32) {
    ret = DoDequantizeInt8ToFp32(int8_ptr_ + thread_offset, float32_ptr_ + thread_offset, quant_arg.scale,
                                 quant_arg.zeroPoint, num_unit_thread);
  } else if (src_dtype == TypeId::kNumberTypeFloat32 && dst_dtype == TypeId::kNumberTypeInt8) {
    bool from_uint8_src = false;
    if (quant_arg.dstDtype == TypeId::kNumberTypeUInt8) {
      from_uint8_src = true;
    }
    ret = DoQuantizeFp32ToInt8(float32_ptr_ + thread_offset, int8_ptr_ + thread_offset, quant_arg.scale,
                               quant_arg.zeroPoint, num_unit_thread, from_uint8_src);
  } else if (src_dtype == TypeId::kNumberTypeInt8 && dst_dtype == TypeId::kNumberTypeUInt8) {
    ret = Int8ToUInt8(int8_ptr_ + thread_offset, uint8_ptr_ + thread_offset, num_unit_thread);
  } else if (src_dtype == TypeId::kNumberTypeUInt8 && dst_dtype == TypeId::kNumberTypeFloat32) {
    ret = DoDequantizeUInt8ToFp32(uint8_ptr_ + thread_offset, float32_ptr_ + thread_offset, quant_arg.scale,
                                  quant_arg.zeroPoint, num_unit_thread);
  } else if (src_dtype == TypeId::kNumberTypeFloat32 && dst_dtype == TypeId::kNumberTypeUInt8) {
    ret = DoQuantizeFp32ToUInt8(float32_ptr_ + thread_offset, uint8_ptr_ + thread_offset, quant_arg.scale,
                                quant_arg.zeroPoint, num_unit_thread);
  } else if (src_dtype == TypeId::kNumberTypeUInt8 && dst_dtype == TypeId::kNumberTypeInt8) {
    ret = UInt8ToInt8(uint8_ptr_ + thread_offset, int8_ptr_ + thread_offset, num_unit_thread);
  } else if (src_dtype == TypeId::kNumberTypeInt8 && dst_dtype == TypeId::kNumberTypeInt8) {
    auto input_quant_arg = in_tensors_.front()->quant_params().front();
    ret = DoDequantizeInt8ToFp32(int8_ptr_ + thread_offset, float32_ptr_ + thread_offset, num_unit_thread,
                                 input_quant_arg.scale, input_quant_arg.zeroPoint);
    if (ret) {
      auto output_quant_arg = out_tensors_.front()->quant_params().front();
      bool from_uint8_src = false;
      if (quant_arg.dstDtype == TypeId::kNumberTypeUInt8) {
        from_uint8_src = true;
      }
      ret = DoQuantizeFp32ToInt8(float32_ptr_ + thread_offset, int8_out_ptr_ + thread_offset, output_quant_arg.scale,
                                 output_quant_arg.zeroPoint, num_unit_thread, from_uint8_src);
    }
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCast error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantDTypeCastRun(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<QuantDTypeCastCPUKernel *>(cdata);
  auto ret = g_kernel->QuantDTypeCast(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCastRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantDTypeCastCPUKernel::Run() {
  if (in_tensors_[0]->data_type() == TypeId::kNumberTypeInt8 &&
      out_tensors_[0]->data_type() == TypeId::kNumberTypeFloat32) {
    int8_ptr_ = reinterpret_cast<int8_t *>(in_tensors_[0]->data_c());
    float32_ptr_ = reinterpret_cast<float *>(out_tensors_[0]->data_c());
  } else if (in_tensors_[0]->data_type() == TypeId::kNumberTypeFloat32 &&
             out_tensors_[0]->data_type() == TypeId::kNumberTypeInt8) {
    float32_ptr_ = reinterpret_cast<float *>(in_tensors_[0]->data_c());
    int8_ptr_ = reinterpret_cast<int8_t *>(out_tensors_[0]->data_c());
  } else if (in_tensors_[0]->data_type() == TypeId::kNumberTypeInt8 &&
             out_tensors_[0]->data_type() == TypeId::kNumberTypeUInt8) {
    int8_ptr_ = reinterpret_cast<int8_t *>(in_tensors_[0]->data_c());
    uint8_ptr_ = reinterpret_cast<uint8_t *>(out_tensors_[0]->data_c());
  } else if (in_tensors_[0]->data_type() == TypeId::kNumberTypeUInt8 &&
             out_tensors_[0]->data_type() == TypeId::kNumberTypeInt8) {
    uint8_ptr_ = reinterpret_cast<uint8_t *>(in_tensors_[0]->data_c());
    int8_ptr_ = reinterpret_cast<int8_t *>(out_tensors_[0]->data_c());
  } else if (in_tensors_[0]->data_type() == TypeId::kNumberTypeInt8 &&
             out_tensors_[0]->data_type() == TypeId::kNumberTypeInt8) {
    int8_ptr_ = reinterpret_cast<int8_t *>(in_tensors_[0]->data_c());
    int8_out_ptr_ = reinterpret_cast<int8_t *>(out_tensors_[0]->data_c());
    float32_ptr_ = new (std::nothrow) float[in_tensors_[0]->ElementsNum()];
    if (float32_ptr_ == nullptr) {
      MS_LOG(ERROR) << "new float[] failed";
      return RET_ERROR;
    }
  } else if (in_tensors_[0]->data_type() == TypeId::kNumberTypeUInt8 &&
             out_tensors_[0]->data_type() == TypeId::kNumberTypeFloat32) {
    uint8_ptr_ = reinterpret_cast<uint8_t *>(in_tensors_[0]->data_c());
    float32_ptr_ = reinterpret_cast<float *>(out_tensors_[0]->data_c());
  } else if (in_tensors_[0]->data_type() == TypeId::kNumberTypeFloat32 &&
             out_tensors_[0]->data_type() == TypeId::kNumberTypeUInt8) {
    float32_ptr_ = reinterpret_cast<float *>(in_tensors_[0]->data_c());
    uint8_ptr_ = reinterpret_cast<uint8_t *>(out_tensors_[0]->data_c());
  }

  auto ret = ParallelLaunch(this->context_->thread_pool_, QuantDTypeCastRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    if (in_tensors_[0]->data_type() == TypeId::kNumberTypeInt8 &&
        out_tensors_[0]->data_type() == TypeId::kNumberTypeInt8) {
      delete (float32_ptr_);
    }
    return RET_ERROR;
  }
  if (in_tensors_[0]->data_type() == TypeId::kNumberTypeInt8 &&
      out_tensors_[0]->data_type() == TypeId::kNumberTypeInt8) {
    delete (float32_ptr_);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeUInt8, PrimitiveType_QuantDTypeCast, LiteKernelCreator<QuantDTypeCastCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_QuantDTypeCast, LiteKernelCreator<QuantDTypeCastCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_QuantDTypeCast, LiteKernelCreator<QuantDTypeCastCPUKernel>)
}  // namespace mindspore::kernel
