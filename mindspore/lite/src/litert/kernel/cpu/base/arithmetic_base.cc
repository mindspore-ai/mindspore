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

#include "src/litert/kernel/cpu/base/arithmetic_base.h"
#include <map>
#include <utility>
#include <vector>
#include "nnacl/base/arithmetic_base.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ArithmeticBaseRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<ArithmeticBaseCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->DoArithmetic(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int ArithmeticBaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (op_parameter_->quant_type_ != schema::QuantType_QUANT_NONE) {
    MS_LOG(ERROR) << "Quant type should be: " << schema::QuantType_QUANT_NONE
                  << " but got: " << op_parameter_->quant_type_;
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->data_type() < kNumberTypeBegin || in_tensors_.at(0)->data_type() > kNumberTypeEnd) {
    MS_LOG(ERROR) << "input0 data_type should be number type but got: " << in_tensors_.at(0)->data_type();
    return RET_ERROR;
  }
  if (in_tensors_.at(1)->data_type() < kNumberTypeBegin || in_tensors_.at(1)->data_type() > kNumberTypeEnd) {
    MS_LOG(ERROR) << "input1 data_type should be number type but got: " << in_tensors_.at(1)->data_type();
    return RET_ERROR;
  }
  primitive_type_ = param_->op_parameter_.type_;
  if (primitive_type_ == schema::PrimitiveType_Eltwise) {
    switch (param_->eltwise_mode_) {
      case schema::EltwiseMode_PROD:
        primitive_type_ = schema::PrimitiveType_MulFusion;
        break;
      case schema::EltwiseMode_SUM:
        primitive_type_ = schema::PrimitiveType_AddFusion;
        break;
      case schema::EltwiseMode_MAXIMUM:
        primitive_type_ = schema::PrimitiveType_Maximum;
        break;
      default:
        MS_LOG(ERROR) << "Eltwise mode not support, mode:" << param_->eltwise_mode_;
        return RET_ERROR;
    }
  }
  InitRunFunction(primitive_type_);
  a_matric_.is_const = in_tensors_[FIRST_INPUT]->IsConst();
  b_matric_.is_const = in_tensors_[SECOND_INPUT]->IsConst();
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticBaseCPUKernel::ReSize() {
  auto ret = ResetStatus();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "reset status failed.";
    return RET_ERROR;
  }
  ret = BroadCastConstTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "broadcast const tensor failed.";
    return ret;
  }
  ComputeOfflineInfo();
  return ChooseThreadCuttingStrategy();
}

int ArithmeticBaseCPUKernel::Run() {
  if (a_matric_.data == nullptr || b_matric_.data == nullptr || c_matric_.data == nullptr) {
    MS_LOG(ERROR) << "exist tensor's data is a nullptr.";
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->ms_context_, ArithmeticBaseRun, this, block_boundary_infos_.size());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "arithmetic failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticBaseCPUKernel::ResetStatus() {
  auto shape0 = in_tensors_[FIRST_INPUT]->shape();
  auto shape1 = in_tensors_[SECOND_INPUT]->shape();
  a_matric_.Reset();
  b_matric_.Reset();
  c_matric_.Reset();
  auto dim_num = shape0.size() >= shape1.size() ? shape0.size() : shape1.size();
  for (size_t i = 0; i < dim_num - shape0.size(); ++i) {
    a_matric_.shape.push_back(1);
  }
  (void)a_matric_.shape.insert(a_matric_.shape.end(), shape0.begin(), shape0.end());
  for (size_t i = 0; i < dim_num - shape1.size(); ++i) {
    b_matric_.shape.push_back(1);
  }
  (void)b_matric_.shape.insert(b_matric_.shape.end(), shape1.begin(), shape1.end());
  auto ret = OptimizeShape();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Optimize shape failed.";
  }
  for (auto buffer : broadcast_buffer_) {
    ms_context_->allocator->Free(buffer);
  }
  broadcast_buffer_.clear();
  block_boundary_infos_.clear();
  return ret;
}

int ArithmeticBaseCPUKernel::OptimizeShape() {
  auto shape0 = a_matric_.shape;
  auto shape1 = b_matric_.shape;
  auto dim_num = shape0.size() >= shape1.size() ? shape0.size() : shape1.size();
  std::vector<int64_t> shape_0;
  std::vector<int64_t> shape_1;
  for (size_t i = 0; i < dim_num - shape0.size(); ++i) {
    shape_0.push_back(1);
  }
  (void)shape_0.insert(shape_0.end(), shape0.begin(), shape0.end());
  for (size_t i = 0; i < dim_num - shape1.size(); ++i) {
    shape_1.push_back(1);
  }
  (void)shape_1.insert(shape_1.end(), shape1.begin(), shape1.end());
  std::vector<int64_t> shape0_temp;
  std::vector<int64_t> shape1_temp;
  for (size_t i = 0; i < dim_num;) {  // horizontal comparison, merge the part of continuous 1.
    shape0_temp.push_back(shape_0[i]);
    shape1_temp.push_back(shape_1[i]);
    if (shape_0[i] != 1 && shape_1[i] != 1) {
      ++i;
      continue;
    }
    size_t j0 = i;
    while (j0 < dim_num && shape_0[j0] == 1) {
      ++j0;
    }
    size_t j1 = i;
    while (j1 < dim_num && shape_1[j1] == 1) {
      ++j1;
    }
    size_t j = MSMAX(j0, j1);
    while ((++i) < j) {
      shape0_temp.back() *= shape_0[i];
      shape1_temp.back() *= shape_1[i];
    }
  }
  shape_0.clear();
  shape_1.clear();
  for (size_t i = 0; i < shape1_temp.size();) {  // vertical comparison, merge the part of continuous equation.
    if (shape0_temp[i] == 1 && shape1_temp[i] == 1) {
      ++i;
      continue;
    }
    shape_0.push_back(shape0_temp[i]);
    shape_1.push_back(shape1_temp[i]);
    if (shape0_temp[i] != shape1_temp[i]) {
      ++i;
      continue;
    }
    while ((++i) < shape0_temp.size()) {
      if (shape0_temp[i] != shape1_temp[i]) {
        break;
      }
      shape_0.back() *= shape0_temp[i];
      shape_1.back() *= shape1_temp[i];
    }
  }
  a_matric_.shape = shape_0;
  b_matric_.shape = shape_1;
  return UpdateParameter();
}

int ArithmeticBaseCPUKernel::UpdateParameter() {
  MS_CHECK_TRUE_MSG(a_matric_.shape.size() == b_matric_.shape.size(), RET_ERROR, "shape-size of a and b is not equal.");
  param_->ndim_ = a_matric_.shape.size();
  MS_CHECK_TRUE_MSG(param_->ndim_ <= ARITHMETIC_SUPPORT_DIMS_NUM, RET_ERROR, "shape-size is out of range.");
  c_matric_.shape.clear();
  for (size_t i = 0; i < param_->ndim_; ++i) {
    MS_CHECK_TRUE_MSG(a_matric_.shape[i] <= INT_MAX, RET_ERROR, "dim is out of int32-max, current not support.");
    MS_CHECK_TRUE_MSG(b_matric_.shape[i] <= INT_MAX, RET_ERROR, "dim is out of int32-max, current not support.");
    param_->in_shape0_[i] = static_cast<int>(a_matric_.shape[i]);
    param_->in_shape1_[i] = static_cast<int>(b_matric_.shape[i]);
    c_matric_.shape.push_back(MSMAX(a_matric_.shape[i], b_matric_.shape[i]));
    param_->out_shape_[i] = MSMAX(param_->in_shape0_[i], param_->in_shape1_[i]);
  }
  return RET_OK;
}

int ArithmeticBaseCPUKernel::BroadCastConstTensor() {
  CalcMultiplesAndStrides(param_);
#ifdef PARALLEL_INFERENCE
  bool prefer_explicit_broadcast = false;
#else
  bool prefer_explicit_broadcast = param_->ndim_ != 1;
#endif
  prefer_explicit_broadcast = prefer_explicit_broadcast && (in_tensors_.front()->data_type() != kNumberTypeBool);
  bool exist_broadcast_{false};
  if (a_matric_.is_const) {
    CHECK_NULL_RETURN(in_tensors_[FIRST_INPUT]->data());
    if (param_->in_elements_num0_ != param_->out_elements_num_ && prefer_explicit_broadcast) {
      a_matric_.data = ms_context_->allocator->Malloc(out_tensors_.front()->ElementsNum() * in_data_size_);
      if (a_matric_.data == nullptr) {
        MS_LOG(ERROR) << "malloc broadcast buffer for input-0 failed";
        return RET_ERROR;
      }
      broadcast_buffer_.push_back(a_matric_.data);
      DoBroadcast(a_matric_.data, FIRST_INPUT);
      param_->in_elements_num0_ = param_->out_elements_num_;
      // shape must be equal to out
      for (size_t i = 0; i < param_->ndim_; ++i) {
        param_->in_shape0_[i] = param_->out_shape_[i];
        param_->in_strides0_[i] = param_->out_strides_[i];
      }
      a_matric_.shape = c_matric_.shape;
      a_matric_.is_valid = true;
      exist_broadcast_ = true;
    }
  }
  if (b_matric_.is_const) {
    CHECK_NULL_RETURN(in_tensors_[SECOND_INPUT]->data());
    if (param_->in_elements_num1_ != param_->out_elements_num_ && prefer_explicit_broadcast) {
      b_matric_.data = ms_context_->allocator->Malloc(out_tensors_.front()->ElementsNum() * in_data_size_);
      if (b_matric_.data == nullptr) {
        MS_LOG(ERROR) << "malloc broadcast buffer for input-1 failed";
        return RET_ERROR;
      }
      broadcast_buffer_.push_back(b_matric_.data);
      DoBroadcast(b_matric_.data, SECOND_INPUT);
      param_->in_elements_num1_ = param_->out_elements_num_;
      // shape must be equal to out
      for (size_t i = 0; i < param_->ndim_; ++i) {
        param_->in_shape1_[i] = param_->out_shape_[i];
        param_->in_strides1_[i] = param_->out_strides_[i];
      }
      b_matric_.shape = c_matric_.shape;
      b_matric_.is_valid = true;
      exist_broadcast_ = true;
    }
  }
  if (!exist_broadcast_) {
    return RET_OK;
  }
  return OptimizeShape();
}

void ArithmeticBaseCPUKernel::ComputeOfflineInfo() {
  int bread_pos{-1};
  int last_dim = static_cast<int>(a_matric_.shape.size()) - 1;
  for (int i = last_dim; i >= 0; --i) {
    if (a_matric_.shape[i] != b_matric_.shape[i]) {
      bread_pos = i;
      break;
    }
  }
  batch_tail_dim_ = bread_pos;
  if (bread_pos == last_dim && batch_tail_dim_ >= 0) {
    --batch_tail_dim_;
  }
  for (int i = last_dim; i > batch_tail_dim_; --i) {
    a_matric_.inner_size *= a_matric_.shape[i];
    b_matric_.inner_size *= b_matric_.shape[i];
    c_matric_.inner_size *= c_matric_.shape[i];
  }
  a_matric_.batch_post_sum = std::vector<int64_t>(a_matric_.shape.size() + 1, 1);
  b_matric_.batch_post_sum = std::vector<int64_t>(b_matric_.shape.size() + 1, 1);
  c_matric_.batch_post_sum = std::vector<int64_t>(c_matric_.shape.size() + 1, 1);
  for (int i = batch_tail_dim_; i >= 0; --i) {
    if (i == batch_tail_dim_) {
      a_matric_.batch_post_sum[i] = a_matric_.shape[i];
      b_matric_.batch_post_sum[i] = b_matric_.shape[i];
      c_matric_.batch_post_sum[i] = c_matric_.shape[i];
    } else {
      a_matric_.batch_post_sum[i] = a_matric_.shape[i] * a_matric_.batch_post_sum[i + 1];
      b_matric_.batch_post_sum[i] = b_matric_.shape[i] * b_matric_.batch_post_sum[i + 1];
      c_matric_.batch_post_sum[i] = c_matric_.shape[i] * c_matric_.batch_post_sum[i + 1];
    }
  }
  if (a_matric_.inner_size == 1) {
    param_->in_elements_num0_ = 1;
    scalar_opt_ = true;
  }
  if (b_matric_.inner_size == 1) {
    param_->in_elements_num1_ = 1;
    scalar_opt_ = true;
  }
}

int ArithmeticBaseCPUKernel::ChooseThreadCuttingStrategy() {
  auto total_num = out_tensors_.front()->ElementsNum();
  if (UpdateThreadNumPass(TC_TYPE(primitive_type_, param_->activation_type_), 1, 1,
                          out_tensors_.at(0)->ElementsNum()) != RET_OK) {
    return RET_ERROR;
  }

  int64_t block_size = UP_DIV(total_num, thread_num_);
  int64_t split_point = 0;
  while (split_point < total_num) {
    int64_t start = split_point;
    int64_t end = start + block_size;
    if (end > total_num) {
      end = total_num;
    }
    BlockBoundaryInfo block_boundary_info;
    block_boundary_info.size_begin = start % c_matric_.inner_size;
    block_boundary_info.size_end = end % c_matric_.inner_size;
    block_boundary_info.batch_begin = start / c_matric_.inner_size;
    block_boundary_info.batch_end = end / c_matric_.inner_size;
    block_boundary_infos_.push_back(block_boundary_info);
    split_point = end;
  }
  return RET_OK;
}

int ArithmeticBaseCPUKernel::DoArithmetic(int task_id) {
  if (block_boundary_infos_[task_id].a_offset.empty()) {
    ComputeOffset(task_id);
  }
  int64_t b_start = block_boundary_infos_[task_id].batch_begin;
  int64_t s_start = block_boundary_infos_[task_id].size_begin;
  int64_t s_end = block_boundary_infos_[task_id].size_end;
  int64_t index_start = 0;
  int64_t index_end = block_boundary_infos_[task_id].batch_end - b_start;
  auto a_ptr = static_cast<uint8_t *>(a_matric_.data) + block_boundary_infos_[task_id].a_offset[index_start];
  auto b_ptr = static_cast<uint8_t *>(b_matric_.data) + block_boundary_infos_[task_id].b_offset[index_start];
  auto c_ptr = static_cast<uint8_t *>(c_matric_.data) + (b_start * c_matric_.inner_size + s_start) * out_data_size_;
  if (a_matric_.inner_size > 1) {
    a_ptr += s_start * in_data_size_;
  }
  if (b_matric_.inner_size > 1) {
    b_ptr += s_start * in_data_size_;
  }
  if (index_start == index_end) {
    auto ret = DoExecute(a_ptr, b_ptr, c_ptr, s_end - s_start);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "failed to calculate.";
      return RET_ERROR;
    }
    return RET_OK;
  }
  int64_t size = c_matric_.inner_size - s_start;
  auto ret = DoExecute(a_ptr, b_ptr, c_ptr, size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to calculate.";
    return RET_ERROR;
  }
  ++index_start;
  c_ptr += size * out_data_size_;
  int64_t c_stride = c_matric_.inner_size * out_data_size_;
  for (; index_start < index_end; ++index_start) {
    a_ptr = static_cast<uint8_t *>(a_matric_.data) + block_boundary_infos_[task_id].a_offset[index_start];
    b_ptr = static_cast<uint8_t *>(b_matric_.data) + block_boundary_infos_[task_id].b_offset[index_start];
    ret = DoExecute(a_ptr, b_ptr, c_ptr, c_matric_.inner_size);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "failed to calculate.";
      return RET_ERROR;
    }
    c_ptr += c_stride;
  }
  if (s_end == 0) {
    return RET_OK;
  }
  a_ptr = static_cast<uint8_t *>(a_matric_.data) + block_boundary_infos_[task_id].a_offset[index_start];
  b_ptr = static_cast<uint8_t *>(b_matric_.data) + block_boundary_infos_[task_id].b_offset[index_start];
  ret = DoExecute(a_ptr, b_ptr, c_ptr, s_end);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to calculate.";
    return RET_ERROR;
  }
  return RET_OK;
}

void ArithmeticBaseCPUKernel::ComputeOffset(int task_id) {
  int64_t b_start = block_boundary_infos_[task_id].batch_begin;
  int64_t b_end = block_boundary_infos_[task_id].batch_end;
  int64_t s_end = block_boundary_infos_[task_id].size_end;
  if (s_end != 0) {
    ++b_end;
  }
  for (; b_start < b_end; ++b_start) {
    int64_t delta = b_start;
    int64_t a_offset = 0;
    int64_t b_offset = 0;
    for (int j = 0; j <= batch_tail_dim_; ++j) {
      if (j > 0) {
        delta = delta % c_matric_.batch_post_sum[j];
      }
      if (j < batch_tail_dim_) {
        a_offset += (delta / c_matric_.batch_post_sum[j + 1] * a_matric_.shape[j] / c_matric_.shape[j]) *
                    a_matric_.batch_post_sum[j + 1];
        b_offset += (delta / c_matric_.batch_post_sum[j + 1] * b_matric_.shape[j] / c_matric_.shape[j]) *
                    b_matric_.batch_post_sum[j + 1];
      } else {
        a_offset += (delta * a_matric_.shape[j] / c_matric_.shape[j]);
        b_offset += (delta * b_matric_.shape[j] / c_matric_.shape[j]);
      }
    }
    block_boundary_infos_[task_id].a_offset.push_back(a_offset * a_matric_.inner_size * in_data_size_);
    block_boundary_infos_[task_id].b_offset.push_back(b_offset * b_matric_.inner_size * in_data_size_);
  }
}
}  // namespace mindspore::kernel
