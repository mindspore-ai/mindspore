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

#include "plugin/device/ascend/kernel/bisheng/bisheng_op_info.h"
#include <memory>
#include <utility>
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "kernel/oplib/op_info_utils.h"
#include "kernel/common_utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::kernel {
BishengOpInfoRegisterHelper::BishengOpInfoRegisterHelper() : op_info_(std::make_shared<OpInfo>()) {
  op_info_->set_imply_type(OpImplyType::kImplyBISHENG);
}

void BishengOpInfoRegisterHelper::OpName(const std::string &name) {
  MS_EXCEPTION_IF_NULL(op_info_);
  op_info_->set_op_name(name);
}

void BishengOpInfoRegisterHelper::Input(size_t index, const std::string &name, bool is_required) {
  MS_EXCEPTION_IF_NULL(op_info_);
  if (inputs_.find(index) != inputs_.end()) {
    MS_LOG(EXCEPTION) << "Repeated input index " << index
                      << (op_info_->op_name().empty() ? "." : " in op " + op_info_->op_name() + ".");
  }
  auto io_info = std::make_shared<OpIOInfo>();
  io_info->set_name(name);
  if (is_required) {
    io_info->set_param_type(kJParamRequred);
  } else {
    io_info->set_param_type(kJParamOptional);
  }
  inputs_.emplace(index, io_info);
  op_info_->add_inputs_ptr(io_info);
}

void BishengOpInfoRegisterHelper::Output(size_t index, const std::string &name, bool is_dynamic) {
  MS_EXCEPTION_IF_NULL(op_info_);
  if (outputs_.find(index) != outputs_.end()) {
    MS_LOG(EXCEPTION) << "Repeated output index " << index
                      << (op_info_->op_name().empty() ? "." : " in op " + op_info_->op_name() + ".");
  }
  auto io_info = std::make_shared<OpIOInfo>();
  io_info->set_name(name);
  if (is_dynamic) {
    io_info->set_param_type(kJParamDynamic);
  } else {
    io_info->set_param_type(kJParamRequred);
  }
  outputs_.emplace(index, io_info);
  op_info_->add_outputs_ptr(io_info);
}

KernelAttr BishengOpInfoRegisterHelper::DataTypeFormat(const std::vector<std::pair<std::string, std::string>> &args) {
  if (args.size() != inputs_.size() + outputs_.size()) {
    MS_LOG(EXCEPTION) << "Invalid dtype&format args size " << args.size() << " with inputs size " << inputs_.size()
                      << " and outputs size " << outputs_.size()
                      << (op_info_->op_name().empty() ? "." : " in op " + op_info_->op_name() + ".");
  }
  MS_EXCEPTION_IF_NULL(op_info_);
  auto kernel_attr = KernelAttr();
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    const auto &data_type = arg.first;
    const auto &format = arg.second;
    if (i < inputs_.size()) {
      auto iter = inputs_.find(i);
      if (iter == inputs_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find input index " << i
                          << (op_info_->op_name().empty() ? "." : " in op " + op_info_->op_name() + ".");
      }
      const auto &io_info = iter->second;
      MS_EXCEPTION_IF_NULL(io_info);
      auto origin_dtypes = io_info->dtypes();
      auto origin_formats = io_info->formats();
      auto origin_object_types = io_info->object_types();
      origin_dtypes.push_back(data_type);
      origin_formats.push_back(format);
      origin_object_types.push_back("tensor");
      io_info->set_dtypes(origin_dtypes);
      io_info->set_formats(origin_formats);
      io_info->set_object_types(origin_object_types);
      kernel_attr.AddInputAttr(tbe::DtypeToTypeId(data_type), format);
    } else {  // is output
      auto iter = outputs_.find(i - inputs_.size());
      if (iter == outputs_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find output index " << i - inputs_.size()
                          << (op_info_->op_name().empty() ? "." : " in op " + op_info_->op_name() + ".");
      }
      const auto &io_info = iter->second;
      MS_EXCEPTION_IF_NULL(io_info);
      auto origin_dtypes = io_info->dtypes();
      auto origin_formats = io_info->formats();
      auto origin_object_types = io_info->object_types();
      origin_dtypes.push_back(data_type);
      origin_formats.push_back(format);
      origin_object_types.push_back("tensor");
      io_info->set_dtypes(origin_dtypes);
      io_info->set_formats(origin_formats);
      io_info->set_object_types(origin_object_types);
      kernel_attr.AddOutputAttr(tbe::DtypeToTypeId(data_type), format);
    }
  }
  return kernel_attr;
}

void BishengOpInfoRegisterHelper::Attr(const std::string &name, const std::string &type, bool is_required) {
  MS_EXCEPTION_IF_NULL(op_info_);
  auto attr_info = std::make_shared<OpAttr>();
  attr_info->set_name(name);
  attr_info->set_type(type);
  if (is_required) {
    attr_info->set_param_type(kJParamRequred);
  } else {
    attr_info->set_param_type(kJParamOptional);
  }
  op_info_->add_attrs_ptr(attr_info);
}

void BishengOpInfoRegisterHelper::End() {
  OpInfoUtils::UpdateRefInfo(op_info_);
  (void)OpLib::GetOpInfoMap()[OpImplyType::kImplyBISHENG].insert(
    std::pair<std::string, OpInfoPtr>(op_info_->op_name() + kImplyBISHENGStr, op_info_));
}
}  // namespace mindspore::kernel
