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
#include "src/runtime/kernel/arm/fp32/transpose_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/pack.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
int TransposeCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TransposeCPUKernel::ReSize() {
  auto &inTensor = in_tensors_.front();
  auto in_shape = inTensor->shape();
  if (in_tensors_.size() == 2) {
    param_->num_axes_ = in_tensors_.at(1)->ElementsNum();
  }
  if (in_shape.size() > MAX_TRANSPOSE_DIM_SIZE) {
    MS_LOG(ERROR) << "input shape out of range.";
    return RET_ERROR;
  }
  int transNd[MAX_TRANSPOSE_DIM_SIZE] = {0, 2, 1};
  int *perm_data = nullptr;
  auto input_tensor = in_tensors_.at(kInputIndex);
  if (input_tensor->shape().size() != static_cast<size_t>(param_->num_axes_)) {
    perm_data = transNd;
    if (input_tensor->shape().size() == C3NUM && param_->num_axes_ == C4NUM) {
      param_->num_axes_ = C3NUM;
    }
    if (param_->num_axes_ == 0) {
      for (int i = 0; i < static_cast<int>(in_shape.size()); ++i) {
        transNd[i] = static_cast<int>(in_shape.size()) - 1 - i;
      }
      param_->num_axes_ = static_cast<int>(in_shape.size());
    }
  } else {
    MS_ASSERT(in_tensors_.size() == C2NUM);
    auto perm_tensor = in_tensors_.at(1);
    perm_data = reinterpret_cast<int *>(perm_tensor->data());
    MSLITE_CHECK_PTR(perm_data);
  }
  MS_CHECK_TRUE_MSG(param_->num_axes_ <= MAX_TRANSPOSE_DIM_SIZE, RET_ERROR, "transpose's perm is invalid.");
  for (int i = 0; i < param_->num_axes_; ++i) {
    param_->perm_[i] = perm_data[i];
  }

  if (GetOptParameters() != RET_OK) {
    MS_LOG(ERROR) << "cannot compute optimizer parameters.";
    return RET_ERROR;
  }
  DecideIfOnlyCopy();
  if (only_copy_) {
    return RET_OK;
  }
  GetOptTransposeFunc();
  if (optTransposeFunc_ != nullptr) {
    return RET_OK;
  }

  auto &outTensor = out_tensors_.front();
  auto out_shape = outTensor->shape();
  param_->strides_[param_->num_axes_ - 1] = 1;
  param_->out_strides_[param_->num_axes_ - 1] = 1;
  param_->data_num_ = inTensor->ElementsNum();
  MS_CHECK_TRUE_RET(static_cast<size_t>(param_->num_axes_) == in_shape.size(), RET_ERROR);
  MS_CHECK_TRUE_RET(static_cast<size_t>(param_->num_axes_) == out_shape.size(), RET_ERROR);
  for (int i = param_->num_axes_ - 2; i >= 0; i--) {
    param_->strides_[i] = in_shape.at(i + 1) * param_->strides_[i + 1];
    param_->out_strides_[i] = out_shape.at(i + 1) * param_->out_strides_[i + 1];
  }

  if (this->out_shape_ != nullptr) {
    free(this->out_shape_);
    this->out_shape_ = nullptr;
  }

  out_shape_ = reinterpret_cast<int *>(malloc(out_shape.size() * sizeof(int)));
  if (out_shape_ == nullptr) {
    MS_LOG(ERROR) << "malloc out_shape_ failed.";
    return RET_ERROR;
  }
  memcpy(out_shape_, out_shape.data(), in_shape.size() * sizeof(int));
  return RET_OK;
}

int TransposeCPUKernel::GetOptParameters() {
  auto in_shape = in_tensors_[0]->shape();
  if (in_shape.size() != static_cast<size_t>(param_->num_axes_)) {
    return RET_OK;
  }
  for (int i = 0; i < param_->num_axes_; i++) {
    if (param_->perm_[i] < 0 || param_->perm_[i] >= param_->num_axes_) {
      MS_LOG(ERROR) << "Check perm failed.";
      return RET_ERROR;
    }
  }
  std::vector<std::vector<int>> segments;
  for (int i = 0; i < param_->num_axes_;) {
    std::vector<int> segment{param_->perm_[i]};
    ++i;
    for (; i < param_->num_axes_; ++i) {
      if (param_->perm_[i] - 1 != param_->perm_[i - 1]) {
        break;
      }
      segment.push_back(param_->perm_[i]);
    }
    segments.push_back(segment);
  }
  in_shape_opt_ = std::vector<int>(segments.size(), 1);
  perm_opt_ = std::vector<int>(segments.size(), 0);
  for (size_t i = 0; i < segments.size(); ++i) {
    for (size_t j = 0; j < segments.size(); ++j) {
      perm_opt_[i] += (segments[j].front() < segments[i].front() ? 1 : 0);
    }
    for (auto index : segments[i]) {
      MS_CHECK_FALSE(INT_MUL_OVERFLOW(in_shape_opt_[perm_opt_[i]], in_shape[index]), RET_ERROR);
      in_shape_opt_[perm_opt_[i]] *= in_shape[index];
    }
  }
  return RET_OK;
}

void TransposeCPUKernel::DecideIfOnlyCopy() {
  auto in_shape = in_tensors_[0]->shape();
  int dim = 0;
  if (in_shape.size() != static_cast<size_t>(param_->num_axes_) || perm_opt_.size() == 1) {
    only_copy_ = true;
    return;
  }
  dim = 0;
  std::vector<int> need_trans_dims;
  std::for_each(perm_opt_.begin(), perm_opt_.end(), [&dim, &need_trans_dims](int val) {
    if (val != dim) {
      need_trans_dims.push_back(dim);
    }
    ++dim;
  });
  if (need_trans_dims.size() == C2NUM && need_trans_dims.back() - need_trans_dims.front() == C1NUM) {
    if (in_shape_opt_[need_trans_dims.front()] == 1 || in_shape_opt_[need_trans_dims.back()] == 1) {
      only_copy_ = true;
      return;
    }
  }
  only_copy_ = false;
}

void TransposeCPUKernel::SetOptTransposeFunc() { optTransposeFunc_ = PackNHWCToNCHWFp32; }

int TransposeCPUKernel::GetOptTransposeFunc() {
  if (in_tensors_[0]->data_type() != kNumberTypeFloat32 || perm_opt_.size() > C3NUM || perm_opt_.size() < C2NUM) {
    optTransposeFunc_ = nullptr;
    return RET_OK;
  }
  bool trans_last_two_dim{true};
  for (size_t i = 0; i < perm_opt_.size() - C2NUM; ++i) {
    if (perm_opt_[i] != static_cast<int>(i)) {
      trans_last_two_dim = false;
      break;
    }
  }
  if (!trans_last_two_dim) {
    optTransposeFunc_ = nullptr;
    return RET_OK;
  }
  SetOptTransposeFunc();
  if (perm_opt_.size() == C2NUM) {
    nhnc_param_[FIRST_INPUT] = 1;
    nhnc_param_[SECOND_INPUT] = in_shape_opt_.front();
    nhnc_param_[THIRD_INPUT] = in_shape_opt_.back();
  } else {
    nhnc_param_[FIRST_INPUT] = in_shape_opt_.front();
    nhnc_param_[SECOND_INPUT] = in_shape_opt_[SECOND_INPUT];
    nhnc_param_[THIRD_INPUT] = in_shape_opt_.back();
  }
  return RET_OK;
}

TransposeCPUKernel::~TransposeCPUKernel() {
  if (this->out_shape_ != nullptr) {
    free(this->out_shape_);
  }
}

int TransposeCPUKernel::TransposeDim2to6() {
  return DoTransposeFp32(static_cast<const float *>(in_data_), static_cast<float *>(out_data_), out_shape_, param_);
}

int TransposeCPUKernel::TransposeDimGreaterThan6(int task_id) {
  TransposeDimsFp32(static_cast<const float *>(in_data_), static_cast<float *>(out_data_), out_shape_, param_, task_id,
                    op_parameter_->thread_num_);
  return RET_OK;
}

int TransposeCPUKernel::CopyInputToOutput() {
  auto in_tensor = in_tensors().front();
  CHECK_NULL_RETURN(in_tensor);
  auto out_tensor = out_tensors().front();
  CHECK_NULL_RETURN(out_tensor);
  if (in_tensor->allocator() == nullptr || in_tensor->allocator() != out_tensor->allocator() ||
      in_tensor->allocator() != ms_context_->allocator || op_parameter_->is_train_session_ ||
      ((in_tensor->IsGraphInput() || in_tensor->IsGraphOutput()) && out_tensor->IsGraphOutput())) {
    CHECK_NULL_RETURN(out_tensor->data());
    CHECK_NULL_RETURN(in_tensor->data());
    MS_CHECK_FALSE(in_tensor->Size() == 0, RET_ERROR);
    if (in_tensor->data() != out_tensor->data()) {
      memcpy(out_tensor->data(), in_tensor->data(), in_tensor->Size());
    }
    return RET_OK;
  }

  out_tensor->FreeData();
  out_tensor->ResetRefCount();
  in_tensor->allocator()->IncRefCount(in_tensor->data(), out_tensor->ref_count());
  out_tensor->set_data(in_tensor->data());
  out_tensor->set_own_data(in_tensor->own_data());
  return RET_OK;
}

int TransposeCPUKernel::RunImpl(int task_id) {
  if (optTransposeFunc_ != nullptr) {
    optTransposeFunc_(in_data_, out_data_, nhnc_param_[FIRST_INPUT], nhnc_param_[SECOND_INPUT],
                      nhnc_param_[THIRD_INPUT], task_id, op_parameter_->thread_num_);
  } else {
    return TransposeDimGreaterThan6(task_id);
  }
  return RET_OK;
}

int TransposeImpl(void *kernel, int task_id, float lhs_scale, float rhs_scale) {
  auto transpose = reinterpret_cast<TransposeCPUKernel *>(kernel);
  auto ret = transpose->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TransposeImpl Run error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int TransposeCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() == 1 || in_tensors_.size() == 2);
  MS_ASSERT(out_tensors_.size() == 1);
  if (only_copy_) {
    return CopyInputToOutput();
  }
  auto &in_tensor = in_tensors_.front();
  auto &out_tensor = out_tensors_.front();
  if (in_tensor == nullptr || out_tensor == nullptr) {
    MS_LOG(ERROR) << "null pointer dreferencing.";
    return RET_ERROR;
  }
  in_data_ = in_tensor->data();
  out_data_ = out_tensor->data();
  CHECK_NULL_RETURN(in_data_);
  CHECK_NULL_RETURN(out_data_);
  if (optTransposeFunc_ != nullptr) {
    return ParallelLaunch(this->ms_context_, TransposeImpl, this, op_parameter_->thread_num_);
  }
  if (out_tensor->shape().size() <= DIMENSION_6D) {
    return TransposeDim2to6();
  } else {
    return ParallelLaunch(this->ms_context_, TransposeImpl, this, op_parameter_->thread_num_);
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Transpose, LiteKernelCreator<TransposeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Transpose, LiteKernelCreator<TransposeCPUKernel>)
}  // namespace mindspore::kernel
