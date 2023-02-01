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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/pattern_selector/tbe_kernel_reduce_selector.h"

#include <string>
#include <vector>
#include <algorithm>
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "kernel/common_utils.h"
#include "common/util/platform_info.h"

namespace mindspore::kernel {
constexpr int64_t kChannelN = 0;
constexpr int64_t kChannelC = 1;
constexpr size_t kReduceNZMinDim = 2;
const int64_t kCubeSize = 16;

void TbeKernelReduceSelector::GetSupportedFormatDType(SupportFormatDType *support_format_dtype) {
  SupportFormat support_format;
  // step1: set ori support
  GetSupportOriFormat(cnode_ptr_, &support_format);
  // step2: get reduce node info
  GetReduceNodeInfo();
  // step3: generate support format
  GetReduceSupport5HD(&support_format);
  GetReduceSupportNDC1HWC0(&support_format);
  GetReduceSupportFracZ(&support_format);
  GetReduceSupportFracNZ(&support_format);
  GetReduceSupportC1HWNCoC0(&support_format);
  GetReduceSupportFracZ3D(&support_format);
  GenerateSupportFormatDType(cnode_ptr_, support_format, support_format_dtype);
  FilterInvalidFormatDType(support_format_dtype);
}

void TbeKernelReduceSelector::GetReduceNodeInfo() {
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode_ptr_);
  auto output_num = AnfAlgo::GetOutputElementNum(cnode_ptr_);
  if (input_num != 1 || output_num != 1) {
    MS_LOG(INFO) << "Reduce operator input/output is not 1, input num: " << input_num << ", output num: " << output_num
                 << ", node info: " << cnode_ptr_->DebugString();
  }
  // get input/output shape
  for (size_t i = 0; i < input_num; ++i) {
    auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode_ptr_, i);
    PadScalarShape(&shape);
    (void)input_shape_.emplace_back(shape);
  }
  for (size_t i = 0; i < output_num; ++i) {
    auto shape = common::AnfAlgo::GetOutputInferShape(cnode_ptr_, i);
    PadScalarShape(&shape);
    (void)output_shape_.emplace_back(shape);
  }
  // get keep dim attr
  GetReduceAttrKeepDim();
  // get axis attr
  axis_ = GetReduceAttrAxis(cnode_ptr_);
  (void)std::transform(axis_.begin(), axis_.end(), axis_.begin(), [&](int64_t elem) {
    if (elem < 0) {
      elem += SizeToLong(input_shape_.at(kIndex0).size());
    }
    return elem;
  });
  GetCheckInfo();
}

void TbeKernelReduceSelector::GetCheckInfo() {
  is_shape_4_dims_ = CheckOriginInputShapeDimEqual(kShape4dDims);
  is_shape_5_dims_ = CheckOriginInputShapeDimEqual(kShape5dDims);
  is_shape_less_2_dims_ = CheckOriginInputShapeDimLess(kShape2dDims);
  is_shape_less_4_dims_ = CheckOriginInputShapeDimLess(kShape4dDims);
  is_shape_less_5_dims_ = CheckOriginInputShapeDimLess(kShape5dDims);
  is_reduce_c_channel_ = CheckReduceContainChanel(kChannelC);
  is_reduce_n_channel_ = CheckReduceContainChanel(kChannelN);
}

void TbeKernelReduceSelector::GetReduceSupport5HD(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  // Note: if input and output size == 2, the last index only support 5hd(float16) --> default
  // keep_dims = false , infer shape error, so not support
  if (!keep_dims_ || !is_shape_4_dims_) {
    return;
  }
  auto support_output_format = is_reduce_c_channel_ ? kOpFormat_DEFAULT : kOpFormat_NC1HWC0;
  GenerateSupportFormat(kOpFormat_NC1HWC0, input_shape_.size(), support_output_format, output_shape_.size(),
                        support_format);
}

void TbeKernelReduceSelector::GetReduceSupportNDC1HWC0(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (!keep_dims_ || !is_shape_5_dims_) {
    return;
  }
  if (is_reduce_c_channel_) {
    return;
  }
  GenerateSupportFormat(kOpFormat_NDC1HWC0, input_shape_.size(), kOpFormat_NDC1HWC0, output_shape_.size(),
                        support_format);
}

void TbeKernelReduceSelector::GetReduceSupportFracZ(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (is_shape_less_4_dims_) {
    return;
  }
  if (!keep_dims_) {
    return;
  }
  if (is_reduce_c_channel_) {
    return;
  }
  if (is_reduce_n_channel_) {
    return;
  }
  GenerateSupportFormat(kOpFormat_FRAC_Z, input_shape_.size(), kOpFormat_FRAC_Z, output_shape_.size(), support_format);
}

void TbeKernelReduceSelector::GetReduceSupportC1HWNCoC0(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (is_shape_less_4_dims_) {
    return;
  }
  if (!keep_dims_) {
    return;
  }
  if (is_reduce_c_channel_) {
    return;
  }
  if (is_reduce_n_channel_) {
    return;
  }
  GenerateSupportFormat(kOpFormat_C1HWNCoC0, input_shape_.size(), kOpFormat_C1HWNCoC0, output_shape_.size(),
                        support_format);
}

void TbeKernelReduceSelector::GetReduceSupportFracZ3D(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (is_shape_less_5_dims_) {
    return;
  }
  if (!keep_dims_) {
    return;
  }
  if (is_reduce_c_channel_) {
    return;
  }
  if (is_reduce_n_channel_) {
    return;
  }
  GenerateSupportFormat(kOpFormat_FRACTAL_Z_3D, input_shape_.size(), kOpFormat_FRACTAL_Z_3D, output_shape_.size(),
                        support_format);
}

void TbeKernelReduceSelector::GetReduceSupportFracNZ(SupportFormat *support_format) const {
  MS_EXCEPTION_IF_NULL(support_format);
  if (is_shape_less_2_dims_) {
    return;
  }
  int64_t last_channel = SizeToLong(input_shape_.at(0).size()) - 1;
  bool is_reduce_last = CheckReduceContainChanel(last_channel);
  int64_t last_channel_but_one = SizeToLong(input_shape_.at(0).size()) - 2;
  bool is_reduce_last_but_one = CheckReduceContainChanel(last_channel_but_one);
  if (!is_reduce_last && !is_reduce_last_but_one) {
    GenerateSupportFormat(kOpFormat_FRAC_NZ, input_shape_.size(), kOpFormat_FRAC_NZ, output_shape_.size(),
                          support_format);
    return;
  }
  if (!keep_dims_) {
    return;
  }
  if (last_channel >= 0 && LongToSize(last_channel) < input_shape_.at(kIndex0).size() &&
      input_shape_.at(kIndex0).at(last_channel) % kCubeSize != 0) {
    return;
  }
  if (last_channel_but_one >= 0 && LongToSize(last_channel_but_one) < input_shape_.at(kIndex0).size() &&
      input_shape_.at(kIndex0).at(last_channel_but_one) % kCubeSize != 0) {
    return;
  }
  GenerateSupportFormat(kOpFormat_FRAC_NZ, input_shape_.size(), kOpFormat_DEFAULT, output_shape_.size(),
                        support_format);
}

void TbeKernelReduceSelector::GetReduceAttrKeepDim() {
  if (!common::AnfAlgo::HasNodeAttr(kAttrKeepDims, cnode_ptr_)) {
    MS_LOG(INFO) << "This node doesn't have keep_attr.";
    keep_dims_ = false;
    return;
  }
  keep_dims_ = common::AnfAlgo::GetNodeAttr<bool>(cnode_ptr_, kAttrKeepDims);
}

void TbeKernelReduceSelector::FilterInvalidFormatDType(SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(support_format_dtype);
  if (support_format_dtype->input_dtypes.size() != 1 || support_format_dtype->output_dtypes.size() != 1) {
    MS_LOG(INFO) << "The reduce node input or output num is not 1.";
    return;
  }

  SupportDTypeItem input_dtypes = support_format_dtype->input_dtypes.at(0);
  SupportFormatItem input_formats = support_format_dtype->input_formats.at(0);
  SupportDTypeItem output_dtypes = support_format_dtype->output_dtypes.at(0);
  SupportFormatItem output_formats = support_format_dtype->output_formats.at(0);

  SupportDTypeItem input_dtypes_new;
  SupportFormatItem input_formats_new;
  SupportDTypeItem output_dtypes_new;
  SupportFormatItem output_formats_new;
  for (size_t i = 0; i < input_dtypes.size(); ++i) {
    auto input_dtype = input_dtypes.at(i);
    auto input_format = input_formats.at(i);
    auto output_dtype = output_dtypes.at(i);
    auto output_format = output_formats.at(i);
    if (input_format == kOpFormat_NC1HWC0 && output_format == kOpFormat_DEFAULT && input_dtype == "float16") {
      MS_LOG(INFO) << "Input 5hd, input type fp16 ane output default not supported.";
      continue;
    }
    (void)input_dtypes_new.emplace_back(input_dtype);
    (void)input_formats_new.emplace_back(input_format);
    (void)output_dtypes_new.emplace_back(output_dtype);
    (void)output_formats_new.emplace_back(output_format);
  }
  support_format_dtype->input_dtypes = {input_dtypes_new};
  support_format_dtype->input_formats = {input_formats_new};
  support_format_dtype->output_dtypes = {output_dtypes_new};
  support_format_dtype->output_formats = {output_formats_new};
}

bool TbeKernelReduceSelector::CheckOriginInputShapeDimEqual(size_t support_dim_size) const {
  // Note: identify format not check
  return std::all_of(input_shape_.begin(), input_shape_.end(),
                     [&support_dim_size](const auto &shape) { return (shape.size() == support_dim_size); });
}

bool TbeKernelReduceSelector::CheckOriginInputShapeDimLess(size_t support_min_dim_size) const {
  // Note: identify format not check
  return std::any_of(input_shape_.begin(), input_shape_.end(),
                     [&support_min_dim_size](const auto &shape) { return (shape.size() < support_min_dim_size); });
}

bool TbeKernelReduceSelector::CheckReduceContainChanel(int64_t channel_index) const {
  // if axis is empty, means reduce all axes, return true;
  if (axis_.empty()) {
    return true;
  }
  // channel out size input size, return true;
  if (channel_index < 0 || input_shape_.at(kIndex0).size() <= LongToSize(channel_index)) {
    return true;
  }
  // any of elem contain channel, return true;
  return std::any_of(axis_.begin(), axis_.end(),
                     [&channel_index](const int64_t &elem) { return (elem == channel_index); });
}
}  // namespace mindspore::kernel
