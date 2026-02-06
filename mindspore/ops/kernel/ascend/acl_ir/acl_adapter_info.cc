/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/acl_ir/acl_adapter_info.h"
#include <algorithm>
#include "utils/ms_context.h"
#include "include/utils/utils.h"
#include "include/utils/pynative/acl_adapter.h"

namespace mindspore::device::ascend {
std::string AclAdapterInfo::SelectFormatFromIndex(size_t index, const std::vector<std::string> &input_formats) const {
  if (output_index_info_.find(index) == output_index_info_.end() ||
      output_index_info_.at(index) >= input_formats.size()) {
    return kOpFormat_DEFAULT;
  }
  return input_formats[output_index_info_.at(index)];
}

std::string AclAdapterInfo::output_format(size_t index, const std::vector<std::string> &input_formats) const {
  if (!output_info_.empty()) {
    const auto &format_list = output_info_.at(index);
    if (format_list.empty()) {
      return SelectFormatFromIndex(index, input_formats);
    }
    auto find = std::find_if(format_list.begin(), format_list.end(), [&input_formats](const auto &format) {
      return std::any_of(input_formats.begin(), input_formats.end(),
                         [&format](const auto &input_format) { return input_format == format; });
    });
    return find == format_list.end() ? kOpFormat_DEFAULT : *find;
  }
  return SelectFormatFromIndex(index, input_formats);
}

AclAdapterManager &AclAdapterManager::GetInstance() {
  static AclAdapterManager instance;
  return instance;
}

AclAdapterInfo &AclAdapterManager::Register(const std::string &op_type) {
  if (op_cache_.count(op_type) != 0) {
    return op_cache_.at(op_type);
  }

  (void)op_cache_.emplace(op_type, AclAdapterInfo(op_type));
  return op_cache_.at(op_type);
}

bool AclAdapterManager::CheckAclAdapter(const std::string &op_type) {
  if (op_cache_.count(op_type) != 0) {
    return true;
  }
  return false;
}

const AclAdapterInfo &AclAdapterManager::GetOpInfo(const std::string &op_type) const {
  if (op_cache_.count(op_type) == 0) {
    MS_LOG(EXCEPTION) << "Current node " << op_type << " hasn't acl adapter";
  }
  return op_cache_.at(op_type);
}
namespace {
std::string GetGraphInfoForAscendSpecial(const std::vector<ValuePtr> &expanded_input_values,
                                         const PrimitivePtr &op_prim, const std::string &graph_info) {
  std::string ascend_special_info = graph_info;
  MS_EXCEPTION_IF_NULL(op_prim);
  auto op_name = op_prim->name();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice &&
      device::ascend::AclAdapterManager::GetInstance().CheckAclAdapter(op_name)) {
    auto acl_info = device::ascend::AclAdapterManager::GetInstance().GetOpInfo(op_name);
    if (!acl_info.input_selector().empty() || acl_info.output_selector() != nullptr) {
      if (expanded_input_values.empty()) {
        return ascend_special_info;
      }
      TypeId first_dtype = TypeId::kTypeUnknown;
      std::vector<ShapeVector> input_shapes;
      (void)std::transform(expanded_input_values.begin(), expanded_input_values.end(), std::back_inserter(input_shapes),
                           [&first_dtype](const ValuePtr &value) -> ShapeVector {
                             auto tensor = value->cast<tensor::TensorPtr>();
                             if (tensor != nullptr) {
                               if (first_dtype == TypeId::kTypeUnknown) {
                                 first_dtype = tensor->data_type();
                               }
                               return tensor->shape();
                             }
                             return {};
                           });

      auto in_func_map = acl_info.input_selector();
      for (auto [index, in_func] : in_func_map) {
        MS_EXCEPTION_IF_NULL(in_func);
        auto tensor = expanded_input_values[index]->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        ascend_special_info += in_func(tensor->data_type(), input_shapes);
      }

      auto out_func = acl_info.output_selector();
      if (out_func != nullptr) {
        auto tensor = expanded_input_values[0]->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        auto out_format = out_func(tensor->data_type(), input_shapes);
        ascend_special_info += out_format;
      }
      MS_EXCEPTION_IF_NULL(out_func);
      auto tensor = expanded_input_values[0]->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto out_format = out_func(tensor->data_type(), input_shapes);
      ascend_special_info += out_format;
    }
  }
  return ascend_special_info;
}

struct AclAdapterCallbackRegister {
  AclAdapterCallbackRegister() noexcept {
    acl_adapter::AclAdapterCallback::SetGetAclGraphInfoFuncHandler(
      [](const std::vector<ValuePtr> &expanded_input_values, const PrimitivePtr &op_prim,
         const std::string &graph_info) -> std::string {
        return GetGraphInfoForAscendSpecial(expanded_input_values, op_prim, graph_info);
      });
  }
} acl_adapter_callback_register;
}  // namespace
}  // namespace  mindspore::device::ascend
