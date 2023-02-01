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
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/pattern_selector/tbe_kernel_broadcast_selector.h"

#include "include/common/utils/utils.h"
#include "utils/trace_base.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"

namespace mindspore::kernel {
constexpr size_t kChannelN = 0;
constexpr size_t kChannelC = 1;
constexpr int64_t kAlignmented16 = 16;
constexpr int64_t kDynamicInputSize = 2;
constexpr int64_t kLastIndex = 1;
constexpr int64_t kLastButOneIndex = 2;

void TbeKernelBroadcastSelector::GetSupportedFormatDType(SupportFormatDType *support_format_dtype) {
  SupportFormat support_format;
  GetSupportOriFormat(cnode_ptr_, &support_format);
  GetBroadCastNodeInfo();
  GetCheckInfo();
  GetBroadcastSupport5HD(&support_format);
  GetBroadcastSupportFracZ(&support_format);
  GetBroadcastSupportC1HWNCoC0(&support_format);
  GetBroadcastSupportFracNZ(&support_format);
  GetBroadcastSupportNDC1HWC0(&support_format);
  GetBroadcastSupportFracZ3D(&support_format);
  GenerateSupportFormatDType(cnode_ptr_, support_format, support_format_dtype);
}

void TbeKernelBroadcastSelector::GetBroadCastNodeInfo() {
  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, cnode_ptr_)) {
    auto dynamic_size_vec = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode_ptr_, kAttrDynInputSizes);
    if (dynamic_size_vec.empty()) {
      MS_LOG(EXCEPTION) << "Node [" << common::AnfAlgo::GetCNodeName(cnode_ptr_)
                        << "]'s attr [dyn_input_sizes] is empty" << trace::DumpSourceLines(cnode_ptr_);
    }
    if (dynamic_size_vec[0] < kDynamicInputSize) {
      MS_LOG(EXCEPTION) << "Node [" << common::AnfAlgo::GetCNodeName(cnode_ptr_)
                        << "]'s attr [dyn_input_sizes] value less than " << kDynamicInputSize
                        << trace::DumpSourceLines(cnode_ptr_);
    }
    auto dynamic_input_shape0_ = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode_ptr_, kIndex0);
    PadScalarShape(&dynamic_input_shape0_);
    (void)input_shapes_.emplace_back(dynamic_input_shape0_);
    input_num_ = 1;
  } else {
    input_num_ = AnfAlgo::GetInputElementNum(cnode_ptr_);
    for (size_t i = 0; i < input_num_; ++i) {
      auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode_ptr_, i);
      PadScalarShape(&input_shape);
      (void)input_shapes_.emplace_back(input_shape);
    }
  }

  output_num_ = AnfAlgo::GetOutputElementNum(cnode_ptr_);
  for (size_t i = 0; i < output_num_; ++i) {
    auto output = common::AnfAlgo::GetOutputInferShape(cnode_ptr_, i);
    PadScalarShape(&output);
    (void)output_shapes_.emplace_back(output);
  }
}

void TbeKernelBroadcastSelector::GetCheckInfo() {
  is_same_shape_ = IsSameShape();
  has_scalar_input_ = HasScalarInput();
}

void TbeKernelBroadcastSelector::GetBroadcastSupport5HD(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  // case1: same shape & no scalar input
  if (is_same_shape_ && !has_scalar_input_) {
    GenerateSupportFormat(kOpFormat_NC1HWC0, input_num_, kOpFormat_NC1HWC0, output_num_, support_format);
    return;
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (has_scalar_input_) {
    // case2: part no scalar input.
    // scalar input: default   |  no scalar input(4D, channel_c % 16 == 0): 5hd
    // scalar output: default  |  no scalar input: 5hd
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        (void)input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (!Is4DShape(shape)) {
          return;
        }
        if (shape[kChannelC] % kAlignmented16 != 0) {
          return;
        }
        (void)input_support_format.emplace_back(kOpFormat_NC1HWC0);
      }
    }
  } else {
    // case3: has no scalar inputs
    // input shape dims == 4 && channel_c can not been broadcast
    for (const auto &shape : input_shapes_) {
      if (!Is4DShape(shape)) {
        return;
      }
      if (IsInputBroadcastByChannel(kChannelC)) {
        MS_LOG(INFO) << "This node broadcast c channel.";
        return;
      }
      input_support_format.assign(input_num_, kOpFormat_NC1HWC0);
    }
  }
  GenOutputSupportFormat(kOpFormat_NC1HWC0, &output_support_format);
  (void)support_format->input_format.emplace_back(input_support_format);
  (void)support_format->output_format.emplace_back(output_support_format);
}

void TbeKernelBroadcastSelector::GetBroadcastSupportFracZ(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (is_same_shape_ && !has_scalar_input_) {
    GenerateSupportFormat(kOpFormat_FRAC_Z, input_num_, kOpFormat_FRAC_Z, output_num_, support_format);
    return;
  }

  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (HasScalarInput()) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        (void)input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (!Is4DShape(shape)) {
          return;
        }
        if (shape[kChannelN] % kAlignmented16 != 0 || shape[kChannelC] % kAlignmented16 != 0) {
          return;
        }
        (void)input_support_format.emplace_back(kOpFormat_FRAC_Z);
      }
    }
  } else {
    // case3: has no scalar inputs
    // input shape dims == 4 && channel_c can not been broadcast
    for (const auto &shape : input_shapes_) {
      if (!Is4DShape(shape)) {
        return;
      }
      if (IsInputBroadcastByChannel(kChannelC) || IsInputBroadcastByChannel(kChannelN)) {
        MS_LOG(INFO) << "This node broadcast c channel.";
        return;
      }
      input_support_format.assign(input_num_, kOpFormat_FRAC_Z);
    }
  }
  GenOutputSupportFormat(kOpFormat_FRAC_Z, &output_support_format);
  (void)support_format->input_format.emplace_back(input_support_format);
  (void)support_format->output_format.emplace_back(output_support_format);
}

void TbeKernelBroadcastSelector::GetBroadcastSupportC1HWNCoC0(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (is_same_shape_ && !has_scalar_input_) {
    GenerateSupportFormat(kOpFormat_C1HWNCoC0, input_num_, kOpFormat_C1HWNCoC0, output_num_, support_format);
    return;
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (has_scalar_input_) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        (void)input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (!Is4DShape(shape)) {
          return;
        }
        if (shape[kChannelN] % kAlignmented16 != 0) {
          return;
        }
        (void)input_support_format.emplace_back(kOpFormat_C1HWNCoC0);
      }
    }
  } else {
    for (const auto &shape : input_shapes_) {
      if (!Is4DShape(shape)) {
        return;
      }
    }
    if (IsInputBroadcastByChannel(kChannelC) || IsInputBroadcastByChannel(kChannelN)) {
      MS_LOG(INFO) << "This node broadcast n || c channel.";
      return;
    }
    input_support_format.assign(input_num_, kOpFormat_C1HWNCoC0);
  }
  GenOutputSupportFormat(kOpFormat_C1HWNCoC0, &output_support_format);
  (void)support_format->input_format.emplace_back(input_support_format);
  (void)support_format->output_format.emplace_back(output_support_format);
}

void TbeKernelBroadcastSelector::GetBroadcastSupportFracNZ(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (is_same_shape_ && !has_scalar_input_) {
    GenerateSupportFormat(kOpFormat_FRAC_NZ, input_num_, kOpFormat_FRAC_NZ, output_num_, support_format);
    return;
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (has_scalar_input_) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        (void)input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (shape.size() < kShape2dDims) {
          return;
        }
        if (shape[shape.size() - kLastIndex] % kAlignmented16 != 0 ||
            shape[shape.size() - kLastButOneIndex] % kAlignmented16 != 0) {
          return;
        }
        (void)input_support_format.emplace_back(kOpFormat_FRAC_NZ);
      }
    }
  } else {
    auto less_2dims = std::any_of(input_shapes_.begin(), input_shapes_.end(),
                                  [](const ShapeVector &elem) { return elem.size() < kShape2dDims; });
    if (less_2dims) {
      MS_LOG(INFO) << "This node dim less 2.";
      return;
    }

    if (IsInputBroadcastByChannel(input_shapes_.front().size() - kLastIndex) ||
        IsInputBroadcastByChannel(input_shapes_.front().size() - kLastButOneIndex)) {
      MS_LOG(INFO) << "This node broadcast last channel.";
      return;
    }

    input_support_format.assign(input_num_, kOpFormat_FRAC_NZ);
  }
  GenOutputSupportFormat(kOpFormat_FRAC_NZ, &output_support_format);
  (void)support_format->input_format.emplace_back(input_support_format);
  (void)support_format->output_format.emplace_back(output_support_format);
}

void TbeKernelBroadcastSelector::GetBroadcastSupportNDC1HWC0(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (is_same_shape_ && !has_scalar_input_) {
    if (input_shapes_.front().size() < kShape5dDims) {
      return;
    }
    GenerateSupportFormat(kOpFormat_NDC1HWC0, input_num_, kOpFormat_NDC1HWC0, output_num_, support_format);
    return;
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (has_scalar_input_) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        (void)input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (!Is5DShape(shape)) {
          return;
        }
        if (shape[kChannelC] % kAlignmented16 != 0) {
          return;
        }
        (void)input_support_format.emplace_back(kOpFormat_NDC1HWC0);
      }
    }
  } else {
    for (const auto &shape : input_shapes_) {
      if (!Is5DShape(shape)) {
        return;
      }
    }
    if (IsInputBroadcastByChannel(kChannelC)) {
      MS_LOG(INFO) << "This node broadcast c channel.";
      return;
    }
    input_support_format.assign(input_num_, kOpFormat_NDC1HWC0);
  }
  GenOutputSupportFormat(kOpFormat_NDC1HWC0, &output_support_format);
  (void)support_format->input_format.emplace_back(input_support_format);
  (void)support_format->output_format.emplace_back(output_support_format);
}

void TbeKernelBroadcastSelector::GetBroadcastSupportFracZ3D(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (is_same_shape_ && !has_scalar_input_) {
    if (input_shapes_.front().size() < kShape5dDims) {
      return;
    }
    GenerateSupportFormat(kOpFormat_FRACTAL_Z_3D, input_num_, kOpFormat_FRACTAL_Z_3D, output_num_, support_format);
    return;
  }
  SupportFormatItem input_support_format;
  SupportFormatItem output_support_format;
  if (has_scalar_input_) {
    for (const auto &shape : input_shapes_) {
      if (IsScalarShape(shape)) {
        (void)input_support_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        if (!Is5DShape(shape)) {
          return;
        }
        if (shape[kChannelC] % kAlignmented16 != 0) {
          return;
        }
        if (shape[kChannelN] % kAlignmented16 != 0) {
          return;
        }
        (void)input_support_format.emplace_back(kOpFormat_FRACTAL_Z_3D);
      }
    }
  } else {
    for (const auto &shape : input_shapes_) {
      if (!Is5DShape(shape)) {
        return;
      }
    }
    if (IsInputBroadcastByChannel(kChannelC) || IsInputBroadcastByChannel(kChannelN)) {
      MS_LOG(INFO) << "This node broadcast c channel.";
      return;
    }
    input_support_format.assign(input_num_, kOpFormat_FRACTAL_Z_3D);
  }
  GenOutputSupportFormat(kOpFormat_FRACTAL_Z_3D, &output_support_format);
  (void)support_format->input_format.emplace_back(input_support_format);
  (void)support_format->output_format.emplace_back(output_support_format);
}

bool TbeKernelBroadcastSelector::Is4DShape(const ShapeVector &shape) { return shape.size() == kShape4dDims; }

bool TbeKernelBroadcastSelector::Is5DShape(const ShapeVector &shape) { return shape.size() == kShape5dDims; }

bool TbeKernelBroadcastSelector::IsSameShape() const {
  auto shape = input_shapes_.begin();
  for (const auto &item : input_shapes_) {
    if (shape->size() != item.size()) {
      return false;
    }
    for (size_t i = 0; i < shape->size(); ++i) {
      if (shape->at(i) != item.at(i) || (shape->at(i) == -1 && item.at(i) == -1)) {
        return false;
      }
    }
  }
  return true;
}

bool TbeKernelBroadcastSelector::IsScalarShape(const ShapeVector &shape) {
  return (shape.size() == 1 && shape[0] == 1);
}

bool TbeKernelBroadcastSelector::HasScalarInput() const {
  return std::any_of(input_shapes_.begin(), input_shapes_.end(),
                     [&](const auto &shape) { return this->IsScalarShape(shape); });
}

bool TbeKernelBroadcastSelector::IsInputBroadcastByChannel(size_t channel) const {
  ShapeValueDType left = 0;
  if (channel < input_shapes_.front().size()) {
    left = input_shapes_.front()[channel];
  }
  return std::any_of(input_shapes_.begin() + 1, input_shapes_.end(), [&left, &channel](const ShapeVector &elem) {
    ShapeValueDType right = 0;
    if (channel < elem.size()) {
      right = elem[channel];
    }
    return (left != right || (left == -1 && right == -1));
  });
}

void TbeKernelBroadcastSelector::GenOutputSupportFormat(const std::string &support_format,
                                                        SupportFormatItem *output_support_item) const {
  MS_EXCEPTION_IF_NULL(output_support_item);
  for (const auto &shape : output_shapes_) {
    if (IsScalarShape(shape)) {
      (void)output_support_item->emplace_back(kOpFormat_DEFAULT);
    } else {
      (void)output_support_item->emplace_back(support_format);
    }
  }
}
}  // namespace mindspore::kernel
