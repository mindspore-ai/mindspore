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
#include "backend/common/optimizer/op_adaptation_info_factory.h"

#include <memory>
#include "kernel/oplib/oplib.h"
#include "utils/log_adapter.h"

namespace mindspore::opt {
OpAdaptationInfoRegister &OpAdaptationInfoRegister::GetInstance() {
  static OpAdaptationInfoRegister inst;
  return inst;
}

std::string OpAdaptationInfoRegister::GenerateKey(const std::string &op_name, const std::string &device_name,
                                                  bool flag) {
  if (device_name != kCPUDevice && device_name != kGPUDevice && device_name != kAscendDevice) {
    MS_LOG(ERROR) << "Backend type is error, " << device_name;
  }

  std::string flag_str = flag ? "true" : "false";
  return std::string(op_name + device_name + flag_str);
}

void OpAdaptationInfoRegister::RegOpAdaptationInfo(OpAdaptationInfo *reg_info) {
  MS_EXCEPTION_IF_NULL(reg_info);
  auto key = GenerateKey(reg_info->GetOriginOpName(), reg_info->GetDeviceName(), reg_info->GetFlag());
  auto find = op_info_map_.find(key);
  if (find != op_info_map_.end()) {
    MS_LOG(ERROR) << "This key (" << key << ")"
                  << " has been registered in origin op info map.";
    return;
  }
  MS_LOG(DEBUG) << "Reg op adaptation info to factory, key: " << key;
  op_info_map_[key] = reg_info;
}

OpAdaptationInfo *OpAdaptationInfoRegister::GetOpAdaptationInfo(const std::string &origin_op_name,
                                                                const std::string &device_name, bool flag) {
  auto key = GenerateKey(origin_op_name, device_name, flag);
  auto iter = op_info_map_.find(key);
  if (iter == op_info_map_.end()) {
    MS_LOG(DEBUG) << "Can't find op adaptation for op " << origin_op_name << " on " << device_name << " when flag is "
                  << flag;
    return nullptr;
  }
  return iter->second;
}

RegisterHelper::RegisterHelper(const string &name, const string &device_name, bool is_dynamic_shape, int len, ...) {
  mindspore::HashSet<size_t> input_to_attr;
  input_to_attr.reserve(static_cast<size_t>(IntToUint(len)));
  va_list var_ptr;
  va_start(var_ptr, len);
  for (int i = 0; i < len; ++i) {
    (void)input_to_attr.insert(static_cast<size_t>(IntToUint(va_arg(var_ptr, int))));
  }
  va_end(var_ptr);
  op_adaptation_info_ = std::make_shared<OpAdaptationInfo>(name, device_name, is_dynamic_shape);
  MS_EXCEPTION_IF_NULL(op_adaptation_info_);
  (void)op_adaptation_info_->SetTargetOpName(name);
  for (auto &index : input_to_attr) {
    (void)op_adaptation_info_->SetInputAttrInfo(index);
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
  origin_op_name_ = op_adaptation_info.origin_op_name_;
  target_op_name_ = op_adaptation_info.target_op_name_;
  pre_check_func_ = op_adaptation_info.pre_check_func_;
  need_tbe_check_supported_ = op_adaptation_info.need_tbe_check_supported_;
  input_attr_map_ = op_adaptation_info.input_attr_map_;
  attr_name_map_ = op_adaptation_info.attr_name_map_;
  device_name_ = op_adaptation_info.device_name_;
  flag_ = op_adaptation_info.flag_;
  return *this;
}
}  // namespace mindspore::opt
