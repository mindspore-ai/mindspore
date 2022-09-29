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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_FACTORY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_FACTORY_H_
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <map>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "include/backend/visible.h"

namespace mindspore::opt {
class InputAttrInfo {
 public:
  explicit InputAttrInfo(const size_t input_index, const std::string attr_name, const std::string attr_data_type)
      : input_index_(input_index), attr_name_(attr_name), attr_data_type_(attr_data_type) {}
  virtual ~InputAttrInfo() = default;

  size_t GetInputIndex() const { return input_index_; }
  std::string GetAttrName() const { return attr_name_; }
  std::string GetAttrDataType() const { return attr_data_type_; }

 private:
  size_t input_index_;
  std::string attr_name_;
  std::string attr_data_type_;
};

class BACKEND_EXPORT OpAdaptationInfo {
 public:
  OpAdaptationInfo(const std::string &origin_op_name, const std::string &device_name, bool flag)
      : origin_op_name_(origin_op_name), target_op_name_(origin_op_name), device_name_(device_name), flag_(flag) {}

  explicit OpAdaptationInfo(const OpAdaptationInfo &op_adaptation_info)
      : origin_op_name_(op_adaptation_info.origin_op_name_),
        target_op_name_(op_adaptation_info.target_op_name_),
        pre_check_func_(op_adaptation_info.pre_check_func_),
        need_tbe_check_supported_(op_adaptation_info.need_tbe_check_supported_),
        input_attr_map_(op_adaptation_info.input_attr_map_),
        attr_name_map_(op_adaptation_info.attr_name_map_),
        device_name_(op_adaptation_info.device_name_),
        flag_(op_adaptation_info.flag_) {}

  OpAdaptationInfo &operator=(const OpAdaptationInfo &op_adaptation_info);
  virtual ~OpAdaptationInfo() = default;

  OpAdaptationInfo &SetTargetOpName(const std::string &target_op_name) {
    target_op_name_ = target_op_name;
    return *this;
  }

  OpAdaptationInfo &SetPreCheckFunc(const std::function<bool(CNodePtr)> &pre_check_func) {
    pre_check_func_ = pre_check_func;
    return *this;
  }

  OpAdaptationInfo &SetNeedTBECheckSupported(bool need_tbe_check_supported) {
    need_tbe_check_supported_ = need_tbe_check_supported;
    return *this;
  }

  OpAdaptationInfo &SetInputAttrInfo(size_t input_index, const std::string &attr_name = "",
                                     const std::string &attr_data_type = "") {
    auto find = input_attr_map_.find(input_index);
    if (find != input_attr_map_.end()) {
      MS_LOG(ERROR) << "This input index (" << input_index << ")"
                    << " has been registered.";
      return *this;
    }
    (void)input_attr_map_.insert(std::make_pair(input_index, InputAttrInfo(input_index, attr_name, attr_data_type)));
    return *this;
  }

  OpAdaptationInfo &SetAttrNameInfo(const std::string &origin_attr_name, const std::string &target_attr_name) {
    if (origin_attr_name.empty() || target_attr_name.empty()) {
      MS_LOG(ERROR) << "Attr name is empty, origin attr name: " << origin_attr_name
                    << ", target attr name:" << target_attr_name << ", origin op name:" << origin_op_name_;
      return *this;
    }

    auto find = attr_name_map_.find(origin_attr_name);
    if (find != attr_name_map_.end()) {
      MS_LOG(ERROR) << "Attr name has been register, origin attr name: " << origin_attr_name
                    << ", old target attr name:" << attr_name_map_[origin_attr_name]
                    << ", new target attr name:" << target_attr_name << ", origin op name:" << origin_op_name_;
    }
    attr_name_map_[origin_attr_name] = target_attr_name;
    return *this;
  }

  std::string GetOriginOpName() const { return origin_op_name_; }
  std::string GetTargetOpName() const { return target_op_name_; }
  std::function<bool(CNodePtr)> GetPreCheckFunc() const { return pre_check_func_; }
  bool NeedTBECheck() const { return need_tbe_check_supported_; }
  std::map<size_t, InputAttrInfo> GetInputAttrInfoMap() const { return input_attr_map_; }
  mindspore::HashMap<std::string, std::string> GetAttrNameInfoMap() const { return attr_name_map_; }
  std::string GetDeviceName() const { return device_name_; }
  bool GetFlag() const { return flag_; }

 private:
  std::string origin_op_name_;
  std::string target_op_name_;
  std::function<bool(CNodePtr)> pre_check_func_{nullptr};
  bool need_tbe_check_supported_{false};
  std::map<size_t, InputAttrInfo> input_attr_map_;
  mindspore::HashMap<std::string, std::string> attr_name_map_;
  std::string device_name_{""};
  bool flag_{false};
};

class BACKEND_EXPORT OpAdaptationInfoRegister {
 public:
  static OpAdaptationInfoRegister &GetInstance();
  static void RegOpAdaptationInfo(OpAdaptationInfo *reg_info);
  [[nodiscard]] static OpAdaptationInfo *GetOpAdaptationInfo(const std::string &origin_op_name,
                                                             const std::string &device_name, bool flag);

 private:
  OpAdaptationInfoRegister() = default;
  ~OpAdaptationInfoRegister() = default;
  DISABLE_COPY_AND_ASSIGN(OpAdaptationInfoRegister)

  static std::string GenerateKey(const std::string &op_name, const std::string &device_name, bool flag);
  // key: (op_name + device_name + flag), value: <OpAdaptationInfo *>
  static std::map<std::string, OpAdaptationInfo *> &GetOpInfoMap();
};

class BACKEND_EXPORT RegisterHelper {
 public:
  RegisterHelper(const std::string &name, const std::string &device_name, bool is_dynamic_shape, int len, ...);
  RegisterHelper(const OpAdaptationInfo &op_adaptation_info);
  ~RegisterHelper() = default;

 private:
  std::shared_ptr<OpAdaptationInfo> op_adaptation_info_{nullptr};
};

#define REG_OP_ADAPTATION_INFO(origin_op_name, device_name, flag)                                      \
  static opt::RegisterHelper g_reg_##device_name##_##origin_op_name##_##flag __attribute__((unused)) = \
    opt::OpAdaptationInfo(origin_op_name, device_name, flag)

#define RER_CONST_TO_ATTR_LIST(origin_op_name, backend, is_dynamic_shape, ...)                                 \
  static opt::RegisterHelper g_reg_##backend##_##is_dynamic_shape##_##origin_op_name(                          \
    origin_op_name, backend, is_dynamic_shape, std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value, \
    __VA_ARGS__)
}  // namespace mindspore::opt
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_FACTORY_H_
