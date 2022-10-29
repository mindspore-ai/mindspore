/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/int8/dynamic_gather_int8.h"
#include <limits>
#include "nnacl/gather_parameter.h"
#include "nnacl/int8/dynamic_gather_int8.h"
#include "nnacl/int8/quantize.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {
DynamicGatherInt8CPUKernel::~DynamicGatherInt8CPUKernel() {
  if (quant_param_ != nullptr) {
    if (quant_param_->zp_in_ != nullptr) {
      free(quant_param_->zp_in_);
      quant_param_->zp_in_ = nullptr;
    }
    if (quant_param_->scale_in_ != nullptr) {
      free(quant_param_->scale_in_);
      quant_param_->scale_in_ = nullptr;
    }
    free(quant_param_);
    quant_param_ = nullptr;
  }
}

int DynamicGatherInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
#ifdef ENABLE_FP16
  enable_fp16_ = ms_context_->device_list_[0].device_info_.cpu_device_info_.enable_float16_;
#endif
  if (in_tensors_.size() == kInputSize2) {
    CHECK_NULL_RETURN(in_tensors_.at(C2NUM));
    auto axis_data = reinterpret_cast<int *>(in_tensors_.at(C2NUM)->data());
    if (axis_data == nullptr) {
      MS_LOG(ERROR) << "DynamicGatherInt8CPUKernel input[2] data nullptr.";
      return RET_ERROR;
    }
    axis_ = *axis_data;
  } else {
    axis_ = (reinterpret_cast<GatherParameter *>(op_parameter_))->axis_;
  }
  auto input_tensor = in_tensors_.at(0);
  if (!input_tensor->IsConst()) {
    MS_LOG(ERROR) << "Does not support tensor0 is non-const.";
    return RET_ERROR;
  }

  auto in_quant_args = input_tensor->quant_params();
  quant_param_ = reinterpret_cast<DynamicGatherQuantArg *>(malloc(sizeof(DynamicGatherQuantArg)));
  if (quant_param_ == nullptr) {
    MS_LOG(ERROR) << "Malloc DynamicGatherQuantArg for dynamic gather int8 op failed!";
    return RET_ERROR;
  }
  memset(quant_param_, 0, sizeof(DynamicGatherQuantArg));
  auto channel_num = in_quant_args.size();
  if (channel_num == 0 || channel_num > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "channel_num must large than 0 and less than 2G.";
    return RET_ERROR;
  }
  quant_param_->scale_in_ = reinterpret_cast<float *>(malloc(channel_num * sizeof(float)));
  CHECK_NULL_RETURN(quant_param_->scale_in_);
  quant_param_->zp_in_ = reinterpret_cast<int32_t *>(malloc(channel_num * sizeof(int32_t)));
  CHECK_NULL_RETURN(quant_param_->zp_in_);
  for (size_t i = 0; i < channel_num; ++i) {
    quant_param_->scale_in_[i] = in_quant_args.at(i).scale;
    quant_param_->zp_in_[i] = in_quant_args.at(i).zeroPoint;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int DynamicGatherInt8CPUKernel::ReSize() {
  // In the framework, the out_tensors data_type is forced to kNumberTypeFloat32
  if (enable_fp16_) {
    out_tensors_[0]->set_data_type(kNumberTypeFloat16);
  }
  auto input_tensor = in_tensors_.at(0);
  auto indices_tensor = in_tensors_.at(1);
  auto in_shape = input_tensor->shape();
  int in_rank = in_shape.size();
  MS_CHECK_LT(axis_, in_rank, RET_ERROR);
  limit_ = in_shape.at(axis_);
  outer_size_ = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_size_ *= in_shape.at(i);
  }
  inner_size_ = 1;
  for (int i = axis_ + 1; i < in_rank; ++i) {
    inner_size_ *= in_shape.at(i);
  }
  indices_element_size_ = indices_tensor->ElementsNum();
  return RET_OK;
}

int DynamicGatherInt8CPUKernel::AssignIndicesData(bool isIndicesInt32, int indices_num, lite::Tensor *indices_tensor,
                                                  int limit) {
  if (!isIndicesInt32) {
    if (indices_num >= std::numeric_limits<int>::max() / static_cast<int>(sizeof(int))) {
      MS_LOG(ERROR) << "Input indices_num is invalid, indices_num: " << indices_num;
      return RET_ERROR;
    }
    indices_data_ = reinterpret_cast<int32_t *>(ms_context_->allocator->Malloc(sizeof(int32_t) * indices_num));
    if (indices_data_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
    switch (indices_tensor->data_type()) {
      case kNumberTypeInt64:
        for (int i = 0; i < indices_num; i++) {
          indices_data_[i] = static_cast<int>(reinterpret_cast<int64_t *>(indices_tensor->MutableData())[i]);
          if (indices_data_[i] >= limit) {
            MS_LOG(ERROR) << " indice data: " << indices_data_[i] << " greater or equal to " << limit;
            return RET_ERROR;
          }
        }
        break;
      case kNumberTypeFloat:
      case kNumberTypeFloat32:
        for (int i = 0; i < indices_num; i++) {
          indices_data_[i] = static_cast<int>(reinterpret_cast<float *>(indices_tensor->MutableData())[i]);
          if (indices_data_[i] >= limit) {
            MS_LOG(ERROR) << " indice data: " << indices_data_[i] << " greater or equal to " << limit;
            return RET_ERROR;
          }
        }
        break;
      default:
        MS_LOG(ERROR) << "Does not support data type: " << indices_tensor->data_type();
        return RET_ERROR;
    }
  } else {
    indices_data_ = reinterpret_cast<int32_t *>(indices_tensor->MutableData());
    for (int i = 0; i < indices_tensor->ElementsNum(); ++i) {
      if (indices_data_[i] >= limit) {
        MS_LOG(ERROR) << " indice data: " << indices_data_[i] << " greater or equal to " << limit;
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int DynamicGatherInt8CPUKernel::DoGather(int task_id) {
  auto input_tensor = in_tensors_.at(0);
  auto indices_tensor = in_tensors_.at(1);
  auto out_tensor = out_tensors_.at(0);

  auto input_ptr = static_cast<int8_t *>(input_tensor->data());
  CHECK_NULL_RETURN(input_ptr);
  float *fp32_output_ptr = nullptr;
#ifdef ENABLE_FP16
  float16_t *fp16_output_ptr = nullptr;
#endif
  if (!enable_fp16_) {
    fp32_output_ptr = static_cast<float *>(out_tensor->data());
    CHECK_NULL_RETURN(fp32_output_ptr);
#ifdef ENABLE_FP16
  } else {
    fp16_output_ptr = static_cast<float16_t *>(out_tensor->data());
    CHECK_NULL_RETURN(fp16_output_ptr);
#endif
  }
  int indices_element_size = indices_tensor->ElementsNum();
  MS_CHECK_GT(indices_element_size, 0, RET_ERROR);

  int stride = UP_DIV(outer_size_, thread_count_);
  int outer_size = MSMIN(stride, outer_size_ - stride * task_id);
  auto thread_stride = stride * task_id;

  input_ptr += thread_stride * inner_size_ * limit_;
  if (!enable_fp16_) {
    fp32_output_ptr += thread_stride * inner_size_ * indices_element_size;
    DynamicGather(input_ptr, outer_size, inner_size_, limit_, indices_data_, indices_element_size_, fp32_output_ptr,
                  quant_param_->scale_in_, quant_param_->zp_in_);
#ifdef ENABLE_FP16
  } else {
    fp16_output_ptr += thread_stride * inner_size_ * indices_element_size;
    DynamicGatherForFp16(input_ptr, outer_size, inner_size_, limit_, indices_data_, indices_element_size_,
                         fp16_output_ptr, quant_param_->scale_in_, quant_param_->zp_in_);
#endif
  }

  return RET_OK;
}

int DynamicGather8Run(void *cdata, int task_id, float, float) {
  auto gather_kernel = reinterpret_cast<DynamicGatherInt8CPUKernel *>(cdata);
  auto error_code = gather_kernel->DoGather(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GatherRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DynamicGatherInt8CPUKernel::Run() {
  auto indices_tensor = in_tensors_.at(1);

  int indices_num = indices_tensor->ElementsNum();
  bool isIndicesInt32 = indices_tensor->data_type() == kNumberTypeInt32;
  int ret = AssignIndicesData(isIndicesInt32, indices_num, indices_tensor, limit_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AssignIndicesData failed, error_code[" << ret << "]";
    return ret;
  }

  int error_code = ParallelLaunch(this->ms_context_, DynamicGather8Run, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Gather function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  if (!isIndicesInt32) {
    ms_context_->allocator->Free(indices_data_);
    indices_data_ = nullptr;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
