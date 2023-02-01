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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/agnostic_selector/tbe_kernel_agnostic_selector.h"

#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"

namespace mindspore::kernel {
void TbeKernelAgnosticSelector::GetSupportedFormatDType(SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(cnode_ptr_);
  SupportFormat support_format;
  auto input_num = AnfAlgo::GetInputElementNum(cnode_ptr_);
  auto output_num = AnfAlgo::GetOutputElementNum(cnode_ptr_);
  if (input_num != 1 || output_num != 1) {
    MS_LOG(EXCEPTION) << "Agnostic only support one input. input_num: " << input_num << ", output num: " << output_num
                      << ", full_name:" << cnode_ptr_->fullname_with_scope();
  }
  auto format = AnfAlgo::GetPrevNodeOutputFormat(cnode_ptr_, 0);
  if (!IsOneOfFormat(format)) {
    MS_LOG(ERROR) << "Got the unknown format " << format;
    return;
  }
  GenerateSupportFormat(format, input_num, format, output_num, &support_format);
  GenerateSupportFormatDType(cnode_ptr_, support_format, support_format_dtype);
}
}  // namespace mindspore::kernel
