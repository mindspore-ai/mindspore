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
#include "src/litert/kernel/cpu/fp32/sparse_to_dense_fp32.h"
#include <vector>
#include <limits>
#include "include/errorcode.h"
#include "nnacl/fp32/sparse_to_dense_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/sparse_to_dense_fp16.h"
#endif
#include "schema/model_generated.h"
#include "schema/ops_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SparseToDense;

namespace mindspore::kernel {
int SparseToDenseCPUKernel::Prepare() {
  MS_CHECK_TRUE_RET(in_tensors_.size() == C4NUM, RET_ERROR);
  CHECK_NULL_RETURN(in_tensors_[FIRST_INPUT]);
  CHECK_NULL_RETURN(in_tensors_[SECOND_INPUT]);
  CHECK_NULL_RETURN(in_tensors_[THIRD_INPUT]);
  CHECK_NULL_RETURN(in_tensors_[FOURTH_INPUT]);
  MS_CHECK_TRUE_RET(out_tensors_.size() == C1NUM, RET_ERROR);
  CHECK_NULL_RETURN(out_tensors_[kOutputIndex]);
  sparse_values_ = in_tensors_[THIRD_INPUT]->data();
  CHECK_NULL_RETURN(sparse_values_);
  param_->is_scalar = in_tensors_[THIRD_INPUT]->ElementsNum() == C1NUM ? true : false;

  default_value_ = in_tensors_[FOURTH_INPUT]->data();
  CHECK_NULL_RETURN(default_value_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SparseToDenseCPUKernel::ReSize() {
  if (in_tensors_.at(THIRD_INPUT)->data_type() != kNumberTypeFloat16 ||
      in_tensors_.at(THIRD_INPUT)->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << in_tensors_.at(THIRD_INPUT)->tensor_name() << " data type "
                  << in_tensors_.at(THIRD_INPUT)->data_type() << " is not support.";
    return RET_ERROR;
  }
  auto output = out_tensors_[kOutputIndex];
  int output_dim = static_cast<int>(output->shape().size());
  MS_CHECK_TRUE_MSG(output_dim <= DIMENSION_4D, RET_ERROR, "output_dim should <= 4");

  int expand_shape[4] = {1, 1, 1, 1};
  for (int i = 0; i < DIMENSION_4D; i++) {
    int pad_dims = DIMENSION_4D - output_dim;
    expand_shape[i] = i >= pad_dims ? output->DimensionSize(i - pad_dims) : 1;
  }
  param_->output_stride[DIMENSION_0D] =
    expand_shape[DIMENSION_1D] * expand_shape[DIMENSION_2D] * expand_shape[DIMENSION_3D];
  param_->output_stride[DIMENSION_1D] = expand_shape[DIMENSION_2D] * expand_shape[DIMENSION_3D];
  param_->output_stride[DIMENSION_2D] = expand_shape[DIMENSION_3D];
  return RET_OK;
}

int SparseToDenseCPUKernel::SetDefaultValue(int task_id) {
  CHECK_NULL_RETURN(indices_vec_);
  CHECK_NULL_RETURN(sparse_values_);
  void *output_data = out_tensors_[kOutputIndex]->data();
  CHECK_NULL_RETURN(output_data);
  int ret = 0;
  if (out_tensors_[kOutputIndex]->data_type() == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    float16_t default_value = *static_cast<float16_t *>(default_value_);
    ret = SparseToDenseSetDefaultFp16(static_cast<float16_t *>(output_data), default_value, param_, task_id);
#endif
  } else {
    float default_value = *static_cast<float *>(default_value_);
    ret = SparseToDenseSetDefault(static_cast<float *>(output_data), default_value, param_, task_id);
  }
  return ret;
}

int SparseToDenseCPUKernel::DoExcute(int task_id) {
  CHECK_NULL_RETURN(indices_vec_);
  CHECK_NULL_RETURN(sparse_values_);
  void *output_data = out_tensors_[kOutputIndex]->data();
  CHECK_NULL_RETURN(output_data);
  int ret = 0;
  if (out_tensors_[kOutputIndex]->data_type() == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    float16_t default_value = *static_cast<float16_t *>(default_value_);
    ret = SparseToDenseFp16(indices_vec_, static_cast<float16_t *>(sparse_values_), default_value,
                            static_cast<float16_t *>(output_data), param_, task_id);
#endif
  } else {
    float default_value = *static_cast<float *>(default_value_);
    ret = SparseToDense(indices_vec_, static_cast<float *>(sparse_values_), default_value,
                        static_cast<float *>(output_data), param_, task_id);
  }
  return ret;
}

int SparseToDenseSetDefaultRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<SparseToDenseCPUKernel *>(cdata);
  auto ret = kernel->SetDefaultValue(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SparseToDenseRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int SparseToDenseRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<SparseToDenseCPUKernel *>(cdata);
  auto ret = kernel->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SparseToDenseRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int SparseToDenseCPUKernel::GenerateIndices() {
  auto input0 = in_tensors_[kInputIndex];
  param_->index_num = input0->DimensionSize(DIMENSION_0D);
  MS_CHECK_TRUE_MSG(param_->index_num != 0, RET_ERROR, "div zero");
  param_->output_num = out_tensors_[kOutputIndex]->ElementsNum();

  if (SIZE_MUL_OVERFLOW(static_cast<size_t>(param_->index_num), sizeof(int) * DIMENSION_4D)) {
    MS_LOG(ERROR) << "Input dim is invalid, dim: " << param_->index_num;
    return RET_ERROR;
  }
  indices_vec_ = static_cast<int *>(
    ms_context_->allocator->Malloc(static_cast<size_t>(param_->index_num) * sizeof(int) * DIMENSION_4D));
  if (indices_vec_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  int *sparse_indices = static_cast<int *>(input0->data());
  CHECK_NULL_RETURN(sparse_indices);

  size_t index_dim = input0->shape().size();
  switch (index_dim) {
    case 0:
    case 1: {
      for (int i = 0; i < param_->index_num; i++) {
        for (int j = 0; j < DIMENSION_4D; j++) {
          indices_vec_[i * DIMENSION_4D + j] = j >= DIMENSION_3D ? sparse_indices[i] : 0;
        }
      }
      break;
    }
    case 2: {
      int real_dims = input0->shape().at(1);
      MS_CHECK_TRUE_MSG(real_dims <= DIMENSION_4D, RET_ERROR, "shape invalid.");
      for (int i = 0; i < param_->index_num; i++) {
        for (int j = 0; j < DIMENSION_4D; j++) {
          int pad_dims = DIMENSION_4D - real_dims;
          indices_vec_[i * DIMENSION_4D + j] = j >= pad_dims ? sparse_indices[i * real_dims + j - pad_dims] : 0;
        }
      }
      break;
    }
    default: {
      MS_LOG(ERROR) << "Indices dimensions is " << index_dim << ", which must be 0, 1 or 2";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int SparseToDenseCPUKernel::Run() {
  auto ret = GenerateIndices();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Generate Indices failed.";
    FreeRunBuff();
    return RET_ERROR;
  }
  // int SparseToDenseSetDefault(float *output, float default_value, SparseToDenseParameter *param, int task_id)
  ret = ParallelLaunch(this->ms_context_, SparseToDenseSetDefaultRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SparseToDenseRun error: error_code[" << ret << "]";
    FreeRunBuff();
    return ret;
  }
  ret = ParallelLaunch(this->ms_context_, SparseToDenseRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SparseToDenseRun error: error_code[" << ret << "]";
  }
  FreeRunBuff();
  return ret;
}

void SparseToDenseCPUKernel::FreeRunBuff() {
  if (indices_vec_ != nullptr) {
    ms_context_->allocator->Free(indices_vec_);
    indices_vec_ = nullptr;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SparseToDense, LiteKernelCreator<SparseToDenseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SparseToDense, LiteKernelCreator<SparseToDenseCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SparseToDense, LiteKernelCreator<SparseToDenseCPUKernel>)
#endif
}  // namespace mindspore::kernel
