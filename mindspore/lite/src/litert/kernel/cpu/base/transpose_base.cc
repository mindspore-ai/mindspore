/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/base/transpose_base.h"
#include <unordered_set>

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
const std::vector<int> kPermOpt = {0, 2, 1};
}  // namespace

int TransposeImpl(void *kernel, int task_id, float lhs_scale, float rhs_scale) {
  auto transpose = reinterpret_cast<TransposeBaseCPUKernel *>(kernel);
  auto ret = transpose->DoTransposeMultiThread(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TransposeImpl Run error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int TransposeBaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TransposeBaseCPUKernel::ReSize() {
  auto ret = ResetStatus();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do transpose reset failed.";
    return ret;
  }
  is_valid_ = in_tensors_[FIRST_INPUT]->shape().size() == static_cast<size_t>(param_->num_axes_);
  if (!is_valid_) {
    return RET_OK;
  }
  ret = OptimizeShape();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do transpose shape-opt failed.";
    return RET_ERROR;
  }
  is_valid_ = perm_.size() > C1NUM;
  if (!is_valid_) {
    return RET_OK;
  }
  opt_run_ = perm_.size() == C2NUM || perm_ == kPermOpt;
  if (opt_run_) {
    SetTransposeOptInfo();
    return RET_OK;
  }
  ret = ComputeOfflineInfo();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do compute transpose offline info failed.";
    return ret;
  }
  return RET_OK;
}

int TransposeBaseCPUKernel::ResetStatus() {
  param_->num_axes_ = 0;
  if (in_tensors_.size() == C2NUM) {
    param_->num_axes_ = in_tensors_[SECOND_INPUT]->ElementsNum();
  }
  auto in_shape = in_tensors_[FIRST_INPUT]->shape();
  if (in_shape.size() > MAX_TRANSPOSE_DIM_SIZE) {
    MS_LOG(ERROR) << "input shape out of range.";
    return RET_ERROR;
  }
  int trans_nd[MAX_TRANSPOSE_DIM_SIZE] = {0, 2, 1};
  int *perm_data{nullptr};
  if (in_shape.size() != static_cast<size_t>(param_->num_axes_)) {
    perm_data = trans_nd;
    if (in_shape.size() == C3NUM && param_->num_axes_ == C4NUM) {
      param_->num_axes_ = C3NUM;
    }
    if (param_->num_axes_ == 0) {
      for (int i = 0; i < static_cast<int>(in_shape.size()); ++i) {
        trans_nd[i] = static_cast<int>(in_shape.size()) - 1 - i;
      }
      param_->num_axes_ = static_cast<int>(in_shape.size());
    }
  } else {
    MS_ASSERT(in_tensors_.size() == C2NUM);
    auto perm_tensor = in_tensors_.at(SECOND_INPUT);
    if (perm_tensor->data_type() != kNumberTypeInt32) {
      MS_LOG(ERROR) << "Unsupported type id: " << perm_tensor->data_type() << " of perm tensor.";
      return RET_ERROR;
    }
    perm_data = reinterpret_cast<int *>(perm_tensor->data());
    MSLITE_CHECK_PTR(perm_data);
    std::vector<int> perm(perm_data, perm_data + in_tensors_[SECOND_INPUT]->ElementsNum());
    if (perm.size() != std::unordered_set<int>(perm.cbegin(), perm.cend()).size()) {
      MS_LOG(ERROR) << "Invalid perm, the same element exits in perm.";
      return RET_ERROR;
    }
  }
  MS_CHECK_TRUE_MSG(param_->num_axes_ <= MAX_TRANSPOSE_DIM_SIZE, RET_ERROR, "transpose perm is invalid.");
  for (int i = 0; i < param_->num_axes_; ++i) {
    param_->perm_[i] = perm_data[i];
  }
  return RET_OK;
}

int TransposeBaseCPUKernel::OptimizeShape() {
  auto in_shape = in_tensors_[0]->shape();
  for (int i = 0; i < param_->num_axes_; i++) {
    if (param_->perm_[i] < 0 || param_->perm_[i] >= param_->num_axes_) {
      MS_LOG(ERROR) << "Check perm failed.";
      return RET_ERROR;
    }
  }
  // first step, delete dimension where value is 1.
  std::vector<int> in_shape_temp;
  std::vector<int> perm_diff(in_shape.size(), 0);
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] != 1) {
      in_shape_temp.push_back(in_shape[i]);
      continue;
    }
    for (size_t j = 0; j < in_shape.size(); ++j) {
      if (param_->perm_[j] < static_cast<int>(i)) {
        continue;
      }
      if (param_->perm_[j] == static_cast<int>(i)) {
        perm_diff[j] = static_cast<int>(i) + 1;
      } else {
        perm_diff[j] += 1;
      }
    }
  }
  std::vector<int> perm_temp;
  for (size_t i = 0; i < in_shape.size(); ++i) {
    int diff = param_->perm_[i] - perm_diff[i];
    if (diff < 0) {
      continue;
    }
    perm_temp.push_back(diff);
  }
  MS_CHECK_TRUE_MSG(in_shape_temp.size() == perm_temp.size(), RET_ERROR, "Do transpose delete dimension failed.");

  // second step, fuse continuous dimension.
  size_t axis_num = in_shape_temp.size();
  std::vector<std::vector<int>> segments;
  for (size_t i = 0; i < axis_num;) {
    std::vector<int> segment{perm_temp[i]};
    ++i;
    for (; i < axis_num; ++i) {
      if (perm_temp[i] - 1 != perm_temp[i - 1]) {
        break;
      }
      segment.push_back(perm_temp[i]);
    }
    segments.push_back(segment);
  }
  in_shape_ = std::vector<int>(segments.size(), 1);
  perm_ = std::vector<int>(segments.size(), 0);
  for (size_t i = 0; i < segments.size(); ++i) {
    for (size_t j = 0; j < segments.size(); ++j) {
      perm_[i] += (segments[j].front() < segments[i].front() ? 1 : 0);
    }
    for (auto index : segments[i]) {
      MS_CHECK_FALSE(INT_MUL_OVERFLOW(in_shape_[perm_[i]], in_shape_temp[index]), RET_ERROR);
      in_shape_[perm_[i]] *= in_shape_temp[index];
    }
  }
  return RET_OK;
}

void TransposeBaseCPUKernel::SetTransposeOptInfo() {
  // now perm is [1, 0] or [0, 2, 1]
  if (perm_.size() == C2NUM) {
    opt_param_[FIRST_INPUT] = 1;
    opt_param_[SECOND_INPUT] = in_shape_.front();
    opt_param_[THIRD_INPUT] = in_shape_.back();
  } else {
    opt_param_[FIRST_INPUT] = in_shape_.front();
    opt_param_[SECOND_INPUT] = in_shape_[SECOND_INPUT];
    opt_param_[THIRD_INPUT] = in_shape_.back();
  }
}

int TransposeBaseCPUKernel::ComputeOfflineInfo() {
  param_->num_axes_ = static_cast<int>(in_shape_.size());
  MS_CHECK_TRUE_MSG(param_->num_axes_ >= C3NUM, RET_ERROR, "The func can run only under axis-num >= 3.");
  for (int i = 0; i < param_->num_axes_; ++i) {
    param_->perm_[i] = perm_[i];
    out_shape_[i] = in_shape_[perm_[i]];
  }
  param_->strides_[param_->num_axes_ - 1] = 1;
  param_->out_strides_[param_->num_axes_ - 1] = 1;
  param_->data_num_ = in_tensors_.front()->ElementsNum();
  for (int i = param_->num_axes_ - 2; i >= 0; i--) {
    param_->strides_[i] = in_shape_[i + 1] * param_->strides_[i + 1];
    param_->out_strides_[i] = out_shape_[i + 1] * param_->out_strides_[i + 1];
  }
  return RET_OK;
}

int TransposeBaseCPUKernel::CopyInputToOutput() {
  auto in_tensor = in_tensors().front();
  CHECK_NULL_RETURN(in_tensor);
  auto out_tensor = out_tensors().front();
  CHECK_NULL_RETURN(out_tensor);
  if (in_tensor->allocator() == nullptr || in_tensor->allocator() != out_tensor->allocator() ||
      in_tensor->allocator() != ms_context_->allocator || op_parameter_->is_train_session_) {
    CHECK_NULL_RETURN(out_tensor->data());
    CHECK_NULL_RETURN(in_tensor->data());
    MS_CHECK_FALSE(in_tensor->Size() == 0, RET_ERROR);
    if (in_tensor->data() != out_tensor->data()) {
      (void)memcpy(out_tensor->data(), in_tensor->data(), in_tensor->Size());
    }
    return RET_OK;
  }

  out_tensor->FreeData();
  out_tensor->ResetRefCount();
  out_tensor->set_data(in_tensor->data());
  if (in_tensor->IsConst()) {
    out_tensor->set_own_data(false);
  } else {
    out_tensor->set_own_data(in_tensor->own_data());
  }
  return RET_OK;
}

int TransposeBaseCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() == C1NUM || in_tensors_.size() == C2NUM);
  MS_ASSERT(out_tensors_.size() == C1NUM);
  if (!is_valid_) {
    return CopyInputToOutput();
  }
  auto &in_tensor = in_tensors_.front();
  auto &out_tensor = out_tensors_.front();
  if (in_tensor == nullptr || out_tensor == nullptr) {
    MS_LOG(ERROR) << "Transpose input-tensor exist nullptr.";
    return RET_ERROR;
  }
  in_data_ = in_tensor->data();
  out_data_ = out_tensor->data();
  CHECK_NULL_RETURN(in_data_);
  CHECK_NULL_RETURN(out_data_);
  if (thread_num_ == 1) {
    return DoTransposeSingleThread();
  }
  return ParallelLaunch(this->ms_context_, TransposeImpl, this, thread_num_);
}
}  // namespace mindspore::kernel
