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
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_selector_creator.h"

#include <map>
#include "kernel/oplib/opinfo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/agnostic_selector/tbe_kernel_agnostic_selector.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/common_selector/tbe_kernel_common_selector.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/dynamic_selector/tbe_kernel_dynamic_selector.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/pattern_selector/tbe_kernel_broadcast_selector.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/pattern_selector/tbe_kernel_reduce_selector.h"

namespace mindspore::kernel {
namespace {
void GetCommonSupportedFormatDType(const CNodePtr &cnode, SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto selector = TbeKernelCommonSelector(cnode);
  selector.GetSupportedFormatDType(support_format_dtype);
}

void GetAgnosticSupportedFormatDType(const CNodePtr &cnode, SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto selector = TbeKernelAgnosticSelector(cnode);
  selector.GetSupportedFormatDType(support_format_dtype);
}

void GetDynamicSupportedFormatDType(const CNodePtr &cnode, SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto selector = TbeKernelDynamicSelector(cnode);
  selector.GetSupportedFormatDType(support_format_dtype);
}

void GetBroadcastSupportedFormatDType(const CNodePtr &cnode, SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto selector = TbeKernelBroadcastSelector(cnode);
  selector.GetSupportedFormatDType(support_format_dtype);
}

void GetReduceSupportedFormatDType(const CNodePtr &cnode, SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto selector = TbeKernelReduceSelector(cnode);
  selector.GetSupportedFormatDType(support_format_dtype);
}

const std::map<OpPattern, GetSupportedFormatDTypeFunc> selector_funcs = {
  {kCommonPattern, GetCommonSupportedFormatDType},
  {kFormatAgnosticPattern, GetAgnosticSupportedFormatDType},
  {kBroadcastPattern, GetBroadcastSupportedFormatDType},
  {kReducePattern, GetReduceSupportedFormatDType},
  {kDynamicFormatPattern, GetDynamicSupportedFormatDType}};
}  // namespace
GetSupportedFormatDTypeFunc GetSelectorFunc(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_info = tbe::TbeDynamicShapeUtil::FindOp(cnode);
  if (!op_info) {
    return nullptr;
  }
  auto pattern = op_info->op_pattern();
  auto iter = selector_funcs.find(pattern);
  return iter == selector_funcs.end() ? nullptr : iter->second;
}
}  // namespace mindspore::kernel
