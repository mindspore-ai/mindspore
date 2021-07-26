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

#include "src/runtime/kernel/arm/base/one_hot_base.h"
#include "nnacl/fp32/one_hot_fp32.h"
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
#include "nnacl/fp16/one_hot_fp16.h"
#endif
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_OneHot;

namespace mindspore::kernel {
namespace {
constexpr size_t kInputNum = 4;
constexpr size_t kInputNumOpt = 3;
constexpr size_t kOutputNum = 1;
}  // namespace

int OneHotCPUKernel::Init() {
  // indices depth on_value off_value
  if ((in_tensors_.size() != kInputNum && in_tensors_.size() != kInputNumOpt) || out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "OneHot input size should be " << kInputNum << " or " << kInputNumOpt << ", got "
                  << in_tensors_.size() << ", output size should be" << kOutputNum << ", got " << out_tensors_.size();
    return RET_ERROR;
  }
  if (ms_context_ == nullptr) {
    MS_LOG(ERROR) << "OneHot context nullptr";
    return RET_NULL_PTR;
  }
  thread_num_ = op_parameter_->thread_num_;

  auto param = reinterpret_cast<OneHotParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "OneHot op_parameter_ nullptr";
    return RET_NULL_PTR;
  }
  axis_ = param->axis_;

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int OneHotCPUKernel::ReSize() {
  auto indices = in_tensors_.at(0);
  if (indices == nullptr) {
    MS_LOG(ERROR) << "OneHot inputs[0] indices nullptr";
    return RET_NULL_PTR;
  }
  auto indices_shape = indices->shape();
  const int indices_rank = static_cast<int>(indices_shape.size());
  if (axis_ < 0) {
    axis_ += indices_rank + 1;
  }

  outer_size_ = 1;
  for (size_t i = 0; i < static_cast<size_t>(axis_); i++) {
    outer_size_ *= indices_shape[i];
  }
  if (outer_size_ == 0) {
    return RET_ERROR;
  }
  inner_size_ = indices->ElementsNum() / outer_size_;

  return RET_OK;
}

int RunOneHot(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto onehot_kernel = reinterpret_cast<OneHotCPUKernel *>(cdata);
  if (onehot_kernel == nullptr) {
    MS_LOG(ERROR) << "cast OneHotCPUKernel failed";
    return RET_ERROR;
  }
  auto error_code = onehot_kernel->OneHotImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "RunOneHot error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int OneHotCPUKernel::OneHotImpl(int task_id) {
  auto indices_data = static_cast<int *>(in_tensors_.at(0)->data());
  if (indices_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto output = out_tensors_.at(0);
  if (output == nullptr) {
    MS_LOG(ERROR) << "OneHot output nullptr";
    return RET_NULL_PTR;
  }
  auto output_data = output->data();
  if (output_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto one_hot_param = reinterpret_cast<OneHotParameter *>(op_parameter_);

  if (output->data_type() == kNumberTypeFloat32) {
    auto ret = OneHotToFp32(indices_data, on_value_, off_value_, reinterpret_cast<float *>(output_data), one_hot_param,
                            task_id, thread_num_);
    return ret;
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  } else if (output->data_type() == kNumberTypeFloat16) {
    auto ret = OneHotToFp16(indices_data, on_value_, off_value_, reinterpret_cast<float16_t *>(output_data),
                            one_hot_param, task_id, thread_num_);
    return ret;
#endif
  } else {
    MS_LOG(ERROR) << "OneHot output datatype is unsupported: " << output->data_type();
    return RET_ERROR;
  }
}

int OneHotCPUKernel::InitParamsAndOnOffValue() {
  auto one_hot_param = reinterpret_cast<OneHotParameter *>(op_parameter_);
  if (one_hot_param == nullptr) {
    MS_LOG(ERROR) << "cast OneHotParameter nullptr";
    return RET_NULL_PTR;
  }

  auto depth_tensor = in_tensors_.at(1);
  if (depth_tensor == nullptr) {
    MS_LOG(ERROR) << "OneHot inputs[1] depth nullptr";
    return RET_NULL_PTR;
  }
  const int *depth = reinterpret_cast<int *>(depth_tensor->MutableData());
  if (depth == nullptr) {
    return RET_NULL_PTR;
  }
  one_hot_param->depth_ = *depth;

  if (in_tensors_.size() == kInputNum) {
    // 4 inputs: indices, depth, on_value, off_value
    one_hot_param->support_neg_index_ = false;
    auto ret = InitOnOffValueForFourInputs();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Init on off value failed";
      return RET_NULL_PTR;
    }
  } else {
    // 3 inputs: indices, depth, off_on_value
    one_hot_param->support_neg_index_ = true;
    auto ret = InitOnOffValueForThreeInputs();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Init on off value failed";
      return RET_NULL_PTR;
    }
  }

  one_hot_param->outer_size_ = outer_size_;
  one_hot_param->inner_size_ = inner_size_;

  return RET_OK;
}

int OneHotCPUKernel::InitOnOffValueForFourInputs() {
  auto on_value_tensor = in_tensors_.at(2);
  if (on_value_tensor == nullptr) {
    MS_LOG(ERROR) << "OneHot on_value tensor is nullptr";
    return RET_NULL_PTR;
  }
  if (on_value_tensor->data_type() == kNumberTypeFloat32) {
    const auto *on_value = reinterpret_cast<float *>(on_value_tensor->data());
    if (on_value == nullptr) {
      return RET_NULL_PTR;
    }
    this->on_value_ = *on_value;
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  } else if (on_value_tensor->data_type() == kNumberTypeFloat16) {
    const auto *on_value = reinterpret_cast<float16_t *>(on_value_tensor->data());
    if (on_value == nullptr) {
      return RET_NULL_PTR;
    }
    this->on_value_ = *on_value;
#endif
  } else {
    MS_LOG(ERROR) << "OneHot on value datatype is unsupported: " << on_value_tensor->data_type();
    return RET_NULL_PTR;
  }

  auto off_value_tensor = in_tensors_.at(3);
  if (off_value_tensor == nullptr) {
    MS_LOG(ERROR) << "OneHot off_value tensor is nullptr";
    return RET_NULL_PTR;
  }

  if (off_value_tensor->data_type() == kNumberTypeFloat32) {
    const auto *off_value = reinterpret_cast<float *>(off_value_tensor->data());
    if (off_value == nullptr) {
      return RET_NULL_PTR;
    }
    this->off_value_ = *off_value;
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  } else if (off_value_tensor->data_type() == kNumberTypeFloat16) {
    const auto *off_value = reinterpret_cast<float16_t *>(off_value_tensor->data());
    if (off_value == nullptr) {
      return RET_NULL_PTR;
    }
    this->off_value_ = *off_value;
#endif
  } else {
    MS_LOG(ERROR) << "OneHot off value datatype is unsupported: " << off_value_tensor->data_type();
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int OneHotCPUKernel::InitOnOffValueForThreeInputs() {
  auto off_on_tensor = in_tensors_.at(2);
  if (off_on_tensor == nullptr) {
    MS_LOG(ERROR) << "OneHot inputs[2] on_value nullptr";
    return RET_NULL_PTR;
  }

  if (off_on_tensor->data_type() == kNumberTypeFloat32) {
    const auto *off_on_values = reinterpret_cast<float *>(off_on_tensor->data());
    if (off_on_values == nullptr) {
      return RET_NULL_PTR;
    }
    this->off_value_ = off_on_values[0];
    this->on_value_ = off_on_values[1];
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  } else if (off_on_tensor->data_type() == kNumberTypeFloat16) {
    const auto *off_on_values = reinterpret_cast<float16_t *>(off_on_tensor->data());
    if (off_on_values == nullptr) {
      return RET_NULL_PTR;
    }
    this->off_value_ = off_on_values[0];
    this->on_value_ = off_on_values[1];
#endif
  } else {
    MS_LOG(ERROR) << "OneHot off value datatype is unsupported: " << off_on_tensor->data_type();
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int OneHotCPUKernel::Run() {
  auto ret = InitParamsAndOnOffValue();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "OneHot init param failed:" << ret;
    return ret;
  }
  int error_code = ParallelLaunch(this->ms_context_, RunOneHot, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "OneHot function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_OneHot, LiteKernelCreator<OneHotCPUKernel>)
}  // namespace mindspore::kernel
