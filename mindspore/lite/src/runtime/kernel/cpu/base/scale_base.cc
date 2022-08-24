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

#include "src/runtime/kernel/cpu/base/scale_base.h"
#include <functional>
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
int CheckInputsOutputsDataType(const std::vector<lite::Tensor *> &in_tensors,
                               const std::vector<lite::Tensor *> &out_tensors) {
  if (std::any_of(in_tensors.begin(), in_tensors.end(), [](const lite::Tensor *input) {
        return input->data_type() != kNumberTypeFloat && input->data_type() != kNumberTypeFloat32 &&
               input->data_type() != kNumberTypeFloat16;
      })) {
    MS_LOG(ERROR) << "Scale: data_type of input should be float32 or float16";
    return RET_ERROR;
  }
  if (std::any_of(out_tensors.begin(), out_tensors.end(), [](const lite::Tensor *output) {
        return output->data_type() != kNumberTypeFloat && output->data_type() != kNumberTypeFloat32 &&
               output->data_type() != kNumberTypeFloat16;
      })) {
    MS_LOG(ERROR) << "Scale: data_type of output should be float32 or float16";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

int ScaleRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto scale = reinterpret_cast<ScaleBaseCPUKernel *>(cdata);
  auto ret = scale->Compute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScaleRun error task_id[" << task_id << "], error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleBaseCPUKernel::Prepare() {
  if (in_tensors_.size() < kInputSize1 || in_tensors_.size() > kInputSize2) {
    MS_LOG(ERROR) << "Scale: inputs-num should be 2 or 3, but now is " << in_tensors_.size();
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto ret = CheckInputsOutputsDataType(in_tensors_, out_tensors_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale: data_type of inputs or output is invalid.";
    return RET_ERROR;
  }
  if (scale_param_->activation_type_ != schema::ActivationType_NO_ACTIVATION &&
      scale_param_->activation_type_ != schema::ActivationType_RELU &&
      scale_param_->activation_type_ != schema::ActivationType_RELU6) {
    MS_LOG(ERROR) << "Scale: activation_type only support relu and relu6, but now is "
                  << scale_param_->activation_type_;
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  ret = ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale: Resize failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleBaseCPUKernel::ReSize() {
  auto ret = CalculateParameter(in_tensors_, scale_param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale: CalculateParameter failed.";
    return RET_ERROR;
  }
  if (InitOffset() != RET_OK) {
    MS_LOG(ERROR) << "Scale: InitOffset when 2-inputs failed.";
    return RET_ERROR;
  }
  if (ComputeThreadCuttingInfo() != RET_OK) {
    MS_LOG(ERROR) << "Scale: compute thread-cutting info failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleBaseCPUKernel::CalculateParameter(const std::vector<lite::Tensor *> &inputs, ScaleParameter *scale_param) {
  auto in_shape = inputs[kInputIndex]->shape();
  auto scale_shape = inputs[kWeightIndex]->shape();
  auto offset_shape = scale_shape;
  if (inputs.size() == kInputSize2) {
    offset_shape = inputs[kBiasIndex]->shape();
  }
  if (!IsValidScale(in_shape, scale_shape, offset_shape, scale_param)) {
    MS_LOG(ERROR) << "Scale: the shape of inputs cannot meet the one of the following specifications.";
    MS_LOG(ERROR) << "    spec1: shapes can be converted into like {[a, b, c], [b, 1] [b, 1]}, can be no Bias.";
    MS_LOG(ERROR) << "    spec1: shapes can be converted into like {[a, b], [b], [b]}, can be no Bias.";
    MS_LOG(ERROR) << "    spec1: shapes can be converted into like {[a, b, c], [b, 1], [1, c]}, can be no Bias.";
    MS_LOG(ERROR) << "    note: axis-attr works on scale and offset in the meantime.";
    return RET_ERROR;
  }

  int outer_size = 1;
  for (int i = 0; i < scale_param->axis_; ++i) {
    outer_size *= in_shape.at(i);
  }
  int axis_with_inner_size = 1;
  for (size_t i = scale_param->axis_; i < in_shape.size(); ++i) {
    axis_with_inner_size *= in_shape[i];
  }
  int axis_size = std::accumulate(scale_shape.begin(), scale_shape.end(), 1, std::multiplies<>());
  scale_param->outer_size_ = outer_size;
  scale_param->axis_size_ = axis_size;
  scale_param->inner_size_ = axis_with_inner_size / axis_size;
  return RET_OK;
}

bool ScaleBaseCPUKernel::IsValidScale(const std::vector<int> &in_shape, const std::vector<int> &scale_shape,
                                      const std::vector<int> &offset_shape, ScaleParameter *scale_param) {
  int axis = scale_param->axis_ < 0 ? scale_param->axis_ + static_cast<int>(in_shape.size()) : scale_param->axis_;
  if (axis < 0 || scale_shape.size() + axis > in_shape.size() || offset_shape.size() + axis > in_shape.size()) {
    MS_LOG(ERROR) << "Scale: axis-attr is invalid, which should be in [0, " << in_shape.size() << "], but now is "
                  << axis;
    return RET_ERROR;
  }
  scale_param->axis_ = axis;
  std::vector<int> expand_scale_shape = scale_shape;
  expand_scale_shape.insert(expand_scale_shape.end(), in_shape.size() - axis - scale_shape.size(), 1);
  std::vector<int> expand_offset_shape = offset_shape;
  expand_offset_shape.insert(expand_offset_shape.end(), in_shape.size() - axis - offset_shape.size(), 1);
  int position_of_diff = 0;
  for (; position_of_diff < static_cast<int>(scale_shape.size()); ++position_of_diff) {
    if (in_shape[position_of_diff + axis] != scale_shape[position_of_diff]) {
      break;
    }
  }
  if (std::any_of(expand_scale_shape.begin() + position_of_diff, expand_scale_shape.end(),
                  [](const int val) { return val != 1; })) {
    MS_LOG(ERROR) << "Scale: the shape of second input is invalid.";
    return false;
  }
  if (expand_scale_shape == expand_offset_shape) {
    scale_param->offset_align_to_axis_ = true;
    return true;
  }
  if (std::any_of(expand_offset_shape.begin(), expand_offset_shape.begin() + position_of_diff,
                  [](const int val) { return val != 1; })) {
    MS_LOG(ERROR) << "Scale: the shape of third input is invalid.";
    return false;
  }
  for (; position_of_diff < static_cast<int>(expand_offset_shape.size()); ++position_of_diff) {
    if (in_shape[position_of_diff + axis] != expand_offset_shape[position_of_diff]) {
      MS_LOG(ERROR) << "Scale: the shape of third input is invalid.";
      return false;
    }
  }
  scale_param->offset_align_to_axis_ = false;
  return true;
}

int ScaleBaseCPUKernel::InitOffset() {
  if (in_tensors_.size() == kInputSize2) {
    return RET_OK;
  }
  if (offset_ != nullptr) {
    ms_context_->allocator->Free(offset_);
  }
  offset_ = ms_context_->allocator->Malloc(scale_param_->axis_size_ * data_type_size_);
  if (offset_ == nullptr) {
    MS_LOG(ERROR) << "Scale: malloc buffer for offset failed.";
    return RET_ERROR;
  }
  memset(offset_, 0, scale_param_->axis_size_ * data_type_size_);
  return RET_OK;
}

int ScaleBaseCPUKernel::ComputeThreadCuttingInfo() {
  split_points_ = {0};
  auto ele_num = out_tensors_.front()->ElementsNum();
  if (UpdateThreadNumPass(TC_TYPE(schema::PrimitiveType_ScaleFusion, scale_param_->activation_type_), 1, 1, ele_num) !=
      RET_OK) {
    MS_LOG(ERROR) << "Scale: cannot determine thread-num.";
    return RET_ERROR;
  }
  if (thread_num_ <= 1) {
    thread_num_ = 1;
    split_points_.push_back(ele_num);
    return RET_OK;
  }
  int64_t block = ele_num / thread_num_;
  int64_t remain = ele_num - block * thread_num_;
  int64_t split = 0;
  while (split < ele_num) {
    split += block;
    split = remain > 0 ? (--remain, split + 1) : split;
    if (split > ele_num) {
      split = ele_num;
    }
    split_points_.push_back(split);
  }
  return RET_OK;
}

int ScaleBaseCPUKernel::Run() {
  if (input_ptr_ == nullptr || scale_ == nullptr || offset_ == nullptr || output_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Scale: find a nullptr in inputs and output.";
    return RET_NULL_PTR;
  }
  auto ret = ParallelLaunch(this->ms_context_, ScaleRun, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
