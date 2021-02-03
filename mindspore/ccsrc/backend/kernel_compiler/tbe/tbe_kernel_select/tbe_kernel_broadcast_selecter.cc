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
#include "backend/kernel_compiler/tbe/tbe_kernel_select/tbe_kernel_broadcast_selecter.h"
#include "utils/utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_select/common_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputIndex_0 = 0;
constexpr size_t kChannelN = 0;
constexpr size_t kChannelC = 1;
constexpr size_t kAlignmented16 = 16;
// 1. all shape no scalar and same
// 2. part scalar : no_scalar (shape size > xxx && alig xxx)
// 3. all no_scalar and not same (broad cast xxx dim)
bool TbeKernelBroadCastSelecter::GetShapeInfo(SupportFormat *support_format) {
  MS_EXCEPTION_IF_NULL(support_format);
  input_num_ = 0;
  output_num_ = 0;
  input_shapes_.clear();
  output_shapes_.clear();
  if (AnfAlgo::HasNodeAttr(kAttrDynInputSizes, cnode_ptr_)) {
    auto dynamic_size_vec = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode_ptr_, kAttrDynInputSizes);
    if (dynamic_size_vec.empty() || dynamic_size_vec[0] < 2) {
      MS_LOG(EXCEPTION) << "dynamic attr set error, please check.";
    }
    auto dynamic_input_shape0_ = AnfAlgo::GetPrevNodeOutputInferShape(cnode_ptr_, kInputIndex_0);
    PadScalarShape(&dynamic_input_shape0_);
    input_shapes_.emplace_back(dynamic_input_shape0_);
    input_num_ = 1;
  } else {
    input_num_ = AnfAlgo::GetInputTensorNum(cnode_ptr_);
    for (size_t i = 0; i < input_num_; ++i) {
      auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode_ptr_, i);
      PadScalarShape(&input_shape);
      input_shapes_.emplace_back(input_shape);
    }
  }

  output_num_ = AnfAlgo::GetOutputTensorNum(cnode_ptr_);
  for (size_t i = 0; i < output_num_; ++i) {
    auto output = AnfAlgo::GetOutputInferShape(cnode_ptr_, i);
    PadScalarShape(&output);
    output_shapes_.emplace_back(output);
  }
  AssignSupportFormat(kOpFormat_DEFAULT, support_format);
  return true;
}

bool TbeKernelBroadCastSelecter::IsBroadCastSupport5HD(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (IsSameShape()) {
    if (!HasScalarInput()) {
      AssignSupportFormat(kOpFormat_NC1HWC0, support_format);
      return true;
    } else {
      return false;
    }
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (HasScalarInput()) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (!Is4DShape(shape)) {
          return false;
        }
        if (shape[kChannelC] % kAlignmented16 != 0) {
          return false;
        }
        input_support_format.emplace_back(kOpFormat_NC1HWC0);
      }
    }
  } else {
    for (const auto &shape : input_shapes_) {
      if (!Is4DShape(shape)) {
        return false;
      }
    }
    auto shape_tmp = input_shapes_[0];
    auto broadcast_c_axis = std::any_of(
      input_shapes_.begin(), input_shapes_.end(),
      [&shape_tmp](const std::vector<size_t> &elem) { return shape_tmp.at(kChannelC) != elem.at(kChannelC); });
    if (broadcast_c_axis) {
      MS_LOG(INFO) << "This node broadcast c channel.";
      return false;
    }
    input_support_format.assign(input_num_, kOpFormat_NC1HWC0);
  }
  GenOutputSupportFormat(kOpFormat_NC1HWC0, &output_support_format);
  support_format->input_format.emplace_back(input_support_format);
  support_format->output_format.emplace_back(output_support_format);
  return true;
}

bool TbeKernelBroadCastSelecter::IsBroadCastSupportFracZ(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (IsSameShape()) {
    if (!HasScalarInput()) {
      AssignSupportFormat(kOpFormat_FRAC_Z, support_format);
      return true;
    } else {
      return false;
    }
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (HasScalarInput()) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (!Is4DShape(shape)) {
          return false;
        }
        if (shape[kChannelN] % kAlignmented16 != 0 || shape[kChannelC] % kAlignmented16 != 0) {
          return false;
        }
        input_support_format.emplace_back(kOpFormat_FRAC_Z);
      }
    }
  } else {
    return false;
  }
  GenOutputSupportFormat(kOpFormat_FRAC_Z, &output_support_format);
  support_format->input_format.emplace_back(input_support_format);
  support_format->output_format.emplace_back(output_support_format);
  return true;
}
bool TbeKernelBroadCastSelecter::IsBroadCastSupportC1HWNCoC0(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (IsSameShape()) {
    if (!HasScalarInput()) {
      AssignSupportFormat(kOpFormat_C1HWNCoC0, support_format);
      return true;
    } else {
      return false;
    }
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (HasScalarInput()) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (!Is4DShape(shape)) {
          return false;
        }
        if (shape[kChannelN] % kAlignmented16 != 0) {
          return false;
        }
        input_support_format.emplace_back(kOpFormat_C1HWNCoC0);
      }
    }
  } else {
    for (const auto &shape : input_shapes_) {
      if (!Is4DShape(shape)) {
        return false;
      }
    }
    auto shape_tmp = input_shapes_[0];
    auto broadcast_nc_axis =
      std::any_of(input_shapes_.begin(), input_shapes_.end(), [&shape_tmp](const std::vector<size_t> &elem) {
        return (shape_tmp.at(kChannelC) != elem.at(kChannelC) || shape_tmp.at(kChannelN) != elem.at(kChannelN));
      });
    if (broadcast_nc_axis) {
      MS_LOG(INFO) << "This node broadcast n || c channel.";
      return false;
    }
    input_support_format.assign(input_num_, kOpFormat_C1HWNCoC0);
  }
  GenOutputSupportFormat(kOpFormat_C1HWNCoC0, &output_support_format);
  support_format->input_format.emplace_back(input_support_format);
  support_format->output_format.emplace_back(output_support_format);
  return true;
}

bool TbeKernelBroadCastSelecter::IsBroadCastSupportFracNZ(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (IsSameShape()) {
    if (!HasScalarInput()) {
      AssignSupportFormat(kOpFormat_FRAC_NZ, support_format);
      return true;
    } else {
      return false;
    }
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (HasScalarInput()) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (shape.size() < kShape2dDims) {
          return false;
        }
        if (shape[shape.size() - 1] % kAlignmented16 != 0 || shape[shape.size() - 2] % kAlignmented16 != 0) {
          return false;
        }
        input_support_format.emplace_back(kOpFormat_FRAC_NZ);
      }
    }
  } else {
    auto less_2dims = std::any_of(input_shapes_.begin(), input_shapes_.end(),
                                  [](const std::vector<size_t> &elem) { return elem.size() < kShape2dDims; });
    if (less_2dims) {
      MS_LOG(INFO) << "This node dim less 2.";
      return false;
    }

    auto shape_tmp = input_shapes_[0];
    auto broadcast_last_dim =
      std::any_of(input_shapes_.begin(), input_shapes_.end(), [&shape_tmp](const std::vector<size_t> &elem) {
        return (shape_tmp.at(shape_tmp.size() - 1) != elem.at(elem.size() - 1)) ||
               (shape_tmp.at(shape_tmp.size() - 2) != elem.at(elem.size() - 2));
      });
    if (broadcast_last_dim) {
      MS_LOG(INFO) << "This node broadcast last channel.";
      return false;
    }

    input_support_format.assign(input_num_, kOpFormat_FRAC_NZ);
  }
  GenOutputSupportFormat(kOpFormat_FRAC_NZ, &output_support_format);
  support_format->input_format.emplace_back(input_support_format);
  support_format->output_format.emplace_back(output_support_format);
  return true;
}

bool TbeKernelBroadCastSelecter::IsBroadCastSupportNDC1HWC0(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (IsSameShape()) {
    if (!HasScalarInput()) {
      AssignSupportFormat(kOpFormat_NDC1HWC0, support_format);
      return true;
    }
    return false;
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (HasScalarInput()) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        input_support_format.emplace_back(kOpFormat_NCDHW);
      } else if (!Is5DShape(shape)) {
        return false;
      } else if (shape[kChannelC] % kAlignmented16 != 0) {
        return false;
      } else {
        input_support_format.emplace_back(kOpFormat_NDC1HWC0);
      }
    }
  } else {
    for (const auto &shape : input_shapes_) {
      if (!Is5DShape(shape)) {
        return false;
      }
    }
    auto shape_tmp = input_shapes_[0];
    auto broadcast_c_axis = std::any_of(
      input_shapes_.begin(), input_shapes_.end(),
      [&shape_tmp](const std::vector<size_t> &elem) { return shape_tmp.at(kChannelC) != elem.at(kChannelC); });
    if (broadcast_c_axis) {
      MS_LOG(INFO) << "This node broadcast c channel.";
      return false;
    }
    input_support_format.assign(input_num_, kOpFormat_NDC1HWC0);
  }
  GenOutputSupportFormat(kOpFormat_NDC1HWC0, &output_support_format);
  support_format->input_format.emplace_back(input_support_format);
  support_format->output_format.emplace_back(output_support_format);
  return true;
}

bool TbeKernelBroadCastSelecter::Is4DShape(const std::vector<size_t> &shape) const {
  return shape.size() == kShape4dDims;
}

bool TbeKernelBroadCastSelecter::Is5DShape(const std::vector<size_t> &shape) const {
  return shape.size() == kShape5dDims;
}

bool TbeKernelBroadCastSelecter::IsSameShape() const {
  auto shape = input_shapes_.begin();
  for (const auto &item : input_shapes_) {
    if (shape->size() != item.size()) {
      return false;
    }
    for (size_t i = 0; i < shape->size(); ++i) {
      if (shape->at(i) != item.at(i)) {
        return false;
      }
    }
  }
  return true;
}

void TbeKernelBroadCastSelecter::PadScalarShape(std::vector<size_t> *shape) const {
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->empty()) {
    shape->emplace_back(1);
  }
}

bool TbeKernelBroadCastSelecter::IsScalarShape(const std::vector<size_t> &shape) const {
  return (shape.size() == 1 && shape[0] == 1);
}

bool TbeKernelBroadCastSelecter::HasScalarInput() const {
  bool ret = false;
  for (const auto &shape : input_shapes_) {
    if (IsScalarShape(shape)) {
      ret = true;
      break;
    }
  }
  return ret;
}

void TbeKernelBroadCastSelecter::GenOutputSupportFormat(const std::string &support_format,
                                                        SupportFormatItem *output_support_item) const {
  MS_EXCEPTION_IF_NULL(output_support_item);
  for (const auto &shape : output_shapes_) {
    if (IsScalarShape(shape)) {
      output_support_item->emplace_back(kOpFormat_DEFAULT);
    } else {
      output_support_item->emplace_back(support_format);
    }
  }
}

void TbeKernelBroadCastSelecter::AssignSupportFormat(const std::string &support_format_str,
                                                     mindspore::kernel::SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  input_support_format.assign(input_num_, support_format_str);
  output_support_format.assign(output_num_, support_format_str);
  support_format->input_format.emplace_back(input_support_format);
  support_format->output_format.emplace_back(output_support_format);
}
}  // namespace kernel
}  // namespace mindspore
