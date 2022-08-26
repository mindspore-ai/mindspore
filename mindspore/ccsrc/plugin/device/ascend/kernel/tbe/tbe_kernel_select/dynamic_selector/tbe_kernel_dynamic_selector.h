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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_DYNAMIC_SELECTOR_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_DYNAMIC_SELECTOR_H_
#include <utility>
#include "ir/anf.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"

namespace mindspore::kernel {
class TbeKernelDynamicSelector {
 public:
  explicit TbeKernelDynamicSelector(CNodePtr cnode_ptr) : cnode_ptr_(std::move(cnode_ptr)) {}
  ~TbeKernelDynamicSelector() = default;
  void GetSupportedFormatDType(SupportFormatDType *support_format_dtype);

 private:
  CNodePtr cnode_ptr_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_DYNAMIC_SELECTOR_H_
