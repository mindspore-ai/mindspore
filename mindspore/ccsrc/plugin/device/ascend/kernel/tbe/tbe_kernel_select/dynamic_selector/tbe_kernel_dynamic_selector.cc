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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/dynamic_selector/tbe_kernel_dynamic_selector.h"

#include <string>
#include <map>
#include <vector>
#include "include/common/utils/utils.h"
#include "include/common/utils/json_operation_utils.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_compile.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"

namespace mindspore::kernel {
constexpr auto kDType1 = "dtype";
constexpr auto kFormat1 = "format";
constexpr auto kUnknownSahpeFormat1 = "unknownshape_format";
constexpr auto kPrefixInput1 = "input";
constexpr auto kPrefixOutput1 = "output";
namespace {
void ParseOpSelectJson(bool is_dynamic_impl, const std::string &op_select_json,
                       SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(support_format_dtype);
  nlohmann::json json_obj;
  if (!ParseJson(op_select_json, &json_obj)) {
    MS_LOG(EXCEPTION) << "Parse op_select_json error.";
  }
  if (!json_obj.is_object()) {
    MS_LOG(EXCEPTION) << "JsonStr is not an object, the json is:" << op_select_json;
  }
  for (const auto &item : json_obj.items()) {
    const std::string &item_name = item.key();
    bool is_input = (item_name.find(kPrefixInput1) != std::string::npos);
    bool is_output = (item_name.find(kPrefixOutput1) != std::string::npos);
    if (!is_input && !is_output) {
      MS_LOG(EXCEPTION) << "Op select ret json is error, the json is:" << op_select_json;
    }
    auto dtypes = SplitStrToVec(item.value().at(kDType1));
    std::string formats_str = item.value().at(kFormat1);
    if (is_dynamic_impl && item.value().find(kUnknownSahpeFormat1) != item.value().end()) {
      formats_str = item.value().at(kUnknownSahpeFormat1);
    }
    auto formats = SplitStrToVec(formats_str);
    if (is_input) {
      (void)support_format_dtype->input_dtypes.emplace_back(dtypes);
      (void)support_format_dtype->input_formats.emplace_back(formats);
    } else {
      (void)support_format_dtype->output_dtypes.emplace_back(dtypes);
      (void)support_format_dtype->output_formats.emplace_back(formats);
    }
  }
}
}  // namespace

void TbeKernelDynamicSelector::GetSupportedFormatDType(SupportFormatDType *support_format_dtype) {
  auto &build_manager = kernel::ascend::TbeKernelCompileManager::GetInstance();
  std::string op_format_dtype_str = build_manager.TbeOpSelectFormat(cnode_ptr_);
  if (op_format_dtype_str.empty()) {
    MS_LOG(EXCEPTION) << "Op select format failed, "
                      << "node name: " << cnode_ptr_->fullname_with_scope();
  }
  auto op_info = tbe::TbeDynamicShapeUtil::FindOp(cnode_ptr_);
  MS_EXCEPTION_IF_NULL(op_info);
  bool is_dynamic_impl = op_info->dynamic_compile_static() || common::AnfAlgo::IsDynamicShape(cnode_ptr_);
  ParseOpSelectJson(is_dynamic_impl, op_format_dtype_str, support_format_dtype);
}
}  // namespace mindspore::kernel
