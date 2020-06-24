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

#include "kernel/tbe/tbe_kernel_select/tbe_kernel_reduce_selecter.h"
#include <string>
#include <vector>
#include "utils/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/tbe/tbe_kernel_select/common_utils.h"

namespace mindspore {
namespace kernel {
constexpr char kKeepDims[] = "keep_dims";
constexpr char kAxis[] = "axis";
constexpr char kTypeInt32[] = "Int32";
constexpr size_t kInputIndex_0 = 0;
constexpr size_t kOutputIndex_0 = 0;
constexpr size_t kChannelN = 0;
constexpr size_t kChannelC = 1;
constexpr size_t kReduceNZMinDim = 3;

bool TbeKernelReduceSelecter::GetShapeInfo(SupportFormat *support_format) {
  MS_EXCEPTION_IF_NULL(support_format);
  input_shape_.clear();
  output_shape_.clear();
  axis_.clear();
  auto input_num = AnfAlgo::GetInputTensorNum(cnode_ptr_);
  auto output_num = AnfAlgo::GetOutputTensorNum(cnode_ptr_);
  if (input_num != 1 || output_num != 1) {
    MS_LOG(EXCEPTION) << "Reduce operator only support one input/output, input num: " << input_num
                      << ", output num: " << output_num;
  }
  // get input/output shape
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(cnode_ptr_, kInputIndex_0);
  PadScalarShape(&input_shape_);
  output_shape_ = AnfAlgo::GetOutputInferShape(cnode_ptr_, kOutputIndex_0);
  PadScalarShape(&output_shape_);
  // get keep dim attr
  GetReduceAttrKeepDim();
  // get axis attr
  GetReduceAttrAxis();
  AssignSupportFormat(kOpFormat_DEFAULT, support_format);
  return true;
}

bool TbeKernelReduceSelecter::IsReduceSupport5HD(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (!Is4DShape(input_shape_)) {
    return false;
  }
  if (!keep_dims_ || axis_.empty()) {
    return false;
  }
  auto reduce_c_axis = std::any_of(axis_.begin(), axis_.end(), [](const size_t &elem) { return (elem == kChannelC); });
  if (reduce_c_axis) {
    return false;
  }
  AssignSupportFormat(kOpFormat_NC1HWC0, support_format);
  return true;
}

bool TbeKernelReduceSelecter::IsReduceSupportNDC1HWC0(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  // like to 5HD
  return false;
}

bool TbeKernelReduceSelecter::IsReduceSupportFracZ(SupportFormat *support_format) const {
  return IsFracZAndC1HWNCoC0Common(kOpFormat_FRAC_Z, support_format);
}

bool TbeKernelReduceSelecter::IsReduceSupportC1HWNCoC0(SupportFormat *support_format) const {
  return IsFracZAndC1HWNCoC0Common(kOpFormat_C1HWNCoC0, support_format);
}

bool TbeKernelReduceSelecter::IsReduceSupportFracNZ(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (input_shape_.size() < kReduceNZMinDim) {
    return false;
  }
  if (axis_.empty()) {
    return false;
  }
  auto reduce_last_axis = std::any_of(axis_.begin(), axis_.end(), [this](const size_t &elem) {
    return (elem == (this->input_shape_.size() - 1) || elem == (this->input_shape_.size() - 2));
  });
  if (reduce_last_axis) {
    return false;
  }
  AssignSupportFormat(kOpFormat_FRAC_NZ, support_format);
  return true;
}

bool TbeKernelReduceSelecter::IsFracZAndC1HWNCoC0Common(const std::string &format,
                                                        mindspore::kernel::SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (!Is4DShape(input_shape_)) {
    return false;
  }
  if (!keep_dims_ || axis_.empty()) {
    return false;
  }
  auto reduce_n_c_axis = std::any_of(axis_.begin(), axis_.end(),
                                     [](const size_t &elem) { return (elem == kChannelC || elem == kChannelN); });
  if (reduce_n_c_axis) {
    return false;
  }
  AssignSupportFormat(format, support_format);
  return true;
}

void TbeKernelReduceSelecter::GetReduceAttrAxis() {
  auto primitive = AnfAlgo::GetCNodePrimitive(cnode_ptr_);
  MS_EXCEPTION_IF_NULL(primitive);
  auto axis = primitive->GetAttr(kAxis);
  if (axis == nullptr) {
    MS_LOG(INFO) << "This node does't have axie attr.";
    return;
  }
  auto type = axis->type();
  MS_EXCEPTION_IF_NULL(type);
  std::vector<int> axis_list;
  if (type->ToString() == kTypeInt32) {
    axis_list.emplace_back(GetValue<int>(axis));
  } else {
    axis_list = GetValue<std::vector<int>>(axis);
  }
  for (const auto &elem : axis_list) {
    if (elem < 0) {
      axis_.emplace_back(input_shape_.size() + elem);
    } else {
      axis_.emplace_back(IntToSize(elem));
    }
  }
}

void TbeKernelReduceSelecter::GetReduceAttrKeepDim() {
  if (!AnfAlgo::HasNodeAttr(kKeepDims, cnode_ptr_)) {
    MS_LOG(INFO) << "This node does't have keep_attr.";
    keep_dims_ = false;
    return;
  }
  keep_dims_ = AnfAlgo::GetNodeAttr<bool>(cnode_ptr_, kKeepDims);
}

void TbeKernelReduceSelecter::AssignSupportFormat(const std::string &support_format_str,
                                                  mindspore::kernel::SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  input_support_format.emplace_back(support_format_str);
  output_support_format.emplace_back(support_format_str);
  support_format->input_format.emplace_back(input_support_format);
  support_format->output_format.emplace_back(output_support_format);
}

bool TbeKernelReduceSelecter::Is4DShape(const std::vector<size_t> &shape) const { return shape.size() == kShape4dDims; }

void TbeKernelReduceSelecter::PadScalarShape(std::vector<size_t> *shape) const {
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->empty()) {
    shape->emplace_back(1);
  }
}
}  // namespace kernel
}  // namespace mindspore
