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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/common_selector/tbe_kernel_common_selector.h"

#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"

namespace mindspore::kernel {
void TbeKernelCommonSelector::GetSupportedFormatDType(SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(support_format_dtype);
  auto op_info = tbe::TbeDynamicShapeUtil::FindOp(cnode_ptr_);
  MS_EXCEPTION_IF_NULL(op_info);
  auto is_dynamic_shape = common::AnfAlgo::IsDynamicShape(cnode_ptr_);
  for (const auto &input : op_info->inputs_ptr()) {
    (void)support_format_dtype->input_dtypes.emplace_back(input->dtypes());
    if (is_dynamic_shape) {
      (void)support_format_dtype->input_formats.emplace_back(input->formats());
    } else {
      (void)support_format_dtype->input_formats.emplace_back(input->formats());
    }
  }
  for (const auto &output : op_info->outputs_ptr()) {
    (void)support_format_dtype->output_dtypes.emplace_back(output->dtypes());
    if (is_dynamic_shape) {
      (void)support_format_dtype->output_formats.emplace_back(output->formats());
    } else {
      (void)support_format_dtype->output_formats.emplace_back(output->formats());
    }
  }
}
}  // namespace mindspore::kernel
