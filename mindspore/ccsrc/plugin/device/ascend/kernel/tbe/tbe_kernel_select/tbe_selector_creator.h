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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_SELECTOR_CREATER_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_SELECTOR_CREATER_
#include <functional>

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"
#include "ir/anf.h"

namespace mindspore::kernel {
using GetSupportedFormatDTypeFunc =
  std::function<void(const CNodePtr &cnode, SupportFormatDType *support_format_dtype)>;
GetSupportedFormatDTypeFunc GetSelectorFunc(const CNodePtr &cnode);
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_SELECTOR_CREATER_
