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
#include "include/backend/optimizer/op_adaptation_info_factory.h"

#include <memory>
#include "kernel/oplib/oplib.h"
#include "utils/log_adapter.h"

namespace mindspore::opt {
OpAdaptationInfo &OpAdaptationInfo::set_backend_op_name(const std::string &default_op_name) {
  backend_op_name_ = default_op_name;
  return *this;
}

OpAdaptationInfo &OpAdaptationInfo::set_target_op_name(const std::string &target_op_name) {
  target_op_name_ = target_op_name;
  return *this;
}

OpAdaptationInfo &OpAdaptationInfo::set_pre_check_func(std::function<bool(CNodePtr)> pre_check_func) {
  pre_check_func_ = std::move(pre_check_func);
  return *this;
}

OpAdaptationInfo &OpAdaptationInfo::set_need_tbe_check_supported(bool need_tbe_check_supported) {
  need_tbe_check_supported_ = need_tbe_check_supported;
  return *this;
}

OpAdaptationInfo &OpAdaptationInfo::set_input_attr_info(size_t input_index, std::string attr_data_type) {
  auto find = input_attr_map_.find(input_index);
  if (find != input_attr_map_.end()) {
    MS_LOG(ERROR) << "This input index (" << input_index << ")"
                  << " has been registered.";
    return *this;
  }
  input_attr_map_[input_index] = attr_data_type;
  return *this;
}

OpAdaptationInfoRegister &OpAdaptationInfoRegister::GetInstance() {
  static OpAdaptationInfoRegister inst;
  return inst;
}

std::string OpAdaptationInfoRegister::GenerateKey(const std::string &me_op_name, const std::string &device_name,
                                                  bool flag) {
  if (device_name != kCPUDevice && device_name != kGPUDevice && device_name != kAscendDevice) {
    MS_LOG(ERROR) << "Backend type is error, " << device_name;
  }

  std::string flag_str = flag ? "true" : "false";
  return std::string(me_op_name + device_name + flag_str);
}

std::map<std::string, OpAdaptationInfo *> &OpAdaptationInfoRegister::GetOpInfoMap() {
  static std::map<std::string, OpAdaptationInfo *> op_info_map;
  return op_info_map;
}

void OpAdaptationInfoRegister::RegOpAdaptationInfo(OpAdaptationInfo *reg_info) {
  MS_EXCEPTION_IF_NULL(reg_info);
  auto key = GenerateKey(reg_info->me_op_name(), reg_info->device_name(), reg_info->flag());
  auto find = GetOpInfoMap().find(key);
  if (find != GetOpInfoMap().end()) {
    MS_LOG(ERROR) << "This key (" << key << ")"
                  << " has been registered in me op info map.";
    return;
  }
  MS_LOG(DEBUG) << "Reg op adaptation info to factory, key: " << key;
  GetOpInfoMap()[key] = reg_info;
}

OpAdaptationInfo *OpAdaptationInfoRegister::GetOpAdaptationInfo(const std::string &me_op_name,
                                                                const std::string &device_name, bool flag) const {
  auto key = GenerateKey(me_op_name, device_name, flag);
  auto iter = GetOpInfoMap().find(key);
  if (iter == GetOpInfoMap().end()) {
    MS_LOG(DEBUG) << "Can't find op adaptation for op " << me_op_name << " on " << device_name << " when flag is "
                  << flag;
    return nullptr;
  }
  return iter->second;
}

RegisterHelper::RegisterHelper(const string &me_op_name, const string &device_name, bool flag, int len, ...) {
  mindspore::HashSet<size_t> input_to_attr;
  input_to_attr.reserve(static_cast<size_t>(IntToUint(len)));
  va_list var_ptr;
  va_start(var_ptr, len);
  for (int i = 0; i < len; ++i) {
    (void)input_to_attr.insert(static_cast<size_t>(IntToUint(va_arg(var_ptr, int))));
  }
  va_end(var_ptr);
  op_adaptation_info_ = std::make_shared<OpAdaptationInfo>(me_op_name, device_name, flag);
  MS_EXCEPTION_IF_NULL(op_adaptation_info_);
  for (auto &index : input_to_attr) {
    (void)op_adaptation_info_->set_input_attr_info(index);
  }
  opt::OpAdaptationInfoRegister::GetInstance().RegOpAdaptationInfo(op_adaptation_info_.get());
}

RegisterHelper::RegisterHelper(const OpAdaptationInfo &op_adaptation_info) {
  op_adaptation_info_ = std::make_shared<OpAdaptationInfo>(op_adaptation_info);
  MS_EXCEPTION_IF_NULL(op_adaptation_info_);
  opt::OpAdaptationInfoRegister::GetInstance().RegOpAdaptationInfo(op_adaptation_info_.get());
}
OpAdaptationInfo &OpAdaptationInfo::operator=(const OpAdaptationInfo &op_adaptation_info) {
  if (this == &op_adaptation_info) {
    return *this;
  }
  me_op_name_ = op_adaptation_info.me_op_name_;
  backend_op_name_ = op_adaptation_info.backend_op_name_;
  target_op_name_ = op_adaptation_info.target_op_name_;
  pre_check_func_ = op_adaptation_info.pre_check_func_;
  need_tbe_check_supported_ = op_adaptation_info.need_tbe_check_supported_;
  input_attr_map_ = op_adaptation_info.input_attr_map_;
  device_name_ = op_adaptation_info.device_name_;
  flag_ = op_adaptation_info.flag_;
  return *this;
}
}  // namespace mindspore::opt
