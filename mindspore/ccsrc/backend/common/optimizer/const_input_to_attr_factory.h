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

class ConvertOpInfo {
 public:
  explicit ConvertOpInfo(const std::string &origin_op_name, const std::string &target_op_name,
                         const std::string &device_name, bool is_dynamic_shape = false)
      : origin_op_name_(origin_op_name),
        target_op_name_(target_op_name),
        device_name_(device_name),
        is_dynamic_shape_(is_dynamic_shape) {}

  explicit ConvertOpInfo(const ConvertOpInfo &convert_op_info)
      : origin_op_name_(convert_op_info.origin_op_name_),
        target_op_name_(convert_op_info.target_op_name_),
        pre_check_func_(convert_op_info.pre_check_func_),
        need_check_supported_(convert_op_info.need_check_supported_),
        input_attr_map_(convert_op_info.input_attr_map_),
        device_name_(convert_op_info.device_name_),
        is_dynamic_shape_(convert_op_info.is_dynamic_shape_) {}

  virtual ~ConvertOpInfo() = default;

  ConvertOpInfo &SetTargetOpName(std::string target_op_name) {
    target_op_name_ = target_op_name;
    return *this;
  }

  ConvertOpInfo &SetPreCheckFunc(std::function<bool(CNodePtr)> pre_check_func) {
    pre_check_func_ = pre_check_func;
    return *this;
  }

  ConvertOpInfo &SetNeedCheckSupported(bool need_check_supported) {
    need_check_supported_ = need_check_supported;
    return *this;
  }

  ConvertOpInfo &SetInputAttrInfo(size_t input_index, std::string attr_name = "", std::string attr_dtype = "") {
    auto find = input_attr_map_.find(input_index);
    if (find != input_attr_map_.end()) {
      MS_LOG(ERROR) << "This input index (" << input_index << ")"
                    << " has been registered.";
      return *this;
    }
    input_attr_map_.insert(std::make_pair(input_index, InputAttrInfo(input_index, attr_name, attr_dtype)));
    return *this;
  }

  std::string GetOriginOpName() const { return origin_op_name_; }
  std::string GetTargetOpName() const { return target_op_name_; }
  std::function<bool(CNodePtr)> GetPreCheckFunc() const { return pre_check_func_; }
  bool GetNeedCheckFlag() const { return need_check_supported_; }
  std::map<size_t, InputAttrInfo> GetInputAttrInfoMap() const { return input_attr_map_; }
  std::string GetDeviceName() const { return device_name_; }
  bool IsDynamicShape() const { return is_dynamic_shape_; }

 private:
  std::string origin_op_name_;
  std::string target_op_name_;
  std::function<bool(CNodePtr)> pre_check_func_{nullptr};
  bool need_check_supported_{false};
  std::map<size_t, InputAttrInfo> input_attr_map_;
  std::string device_name_;
  bool is_dynamic_shape_{false};
};

class BACKEND_EXPORT ConvertOpInfoRegister {
 public:
  static ConvertOpInfoRegister &GetInstance();
  void RegConvertOpInfo(ConvertOpInfo *reg_info);
  [[nodiscard]] ConvertOpInfo *GetConvertOpInfo(const std::string &origin_op_name, const std::string &device_name,
                                                bool is_dynamic_shape) const;

 private:
  ConvertOpInfoRegister() = default;
  ~ConvertOpInfoRegister() = default;
  DISABLE_COPY_AND_ASSIGN(ConvertOpInfoRegister)

  static std::string GenerateKey(const std::string &op_name, const std::string &device_name, bool is_dynamic_shape);
  // key: (op_name + device_name + is_dynamic), value: <ConvertOpInfo *>
  std::map<std::string, ConvertOpInfo *> op_info_map_;
};

class RegisterHelper {
 public:
  RegisterHelper(const std::string &name, const std::string &device_name, bool is_dynamic_shape, int len, ...);
  explicit RegisterHelper(const ConvertOpInfo &convert_op_info);
  ~RegisterHelper() = default;

 private:
  std::shared_ptr<ConvertOpInfo> convert_op_info_{nullptr};
};

#define REG_CONST_TO_ATTR(origin_op_name, target_op_name, device_name, dynamic)                    \
  static opt::RegisterHelper g_reg_##device_name##_##dynamic##_##origin_op_name##_##target_op_name \
    __attribute__((unused)) = opt::ConvertOpInfo(origin_op_name, target_op_name, device_name, dynamic)

#define RER_CONST_TO_ATTR_LIST(origin_op_name, backend, dynamic, ...)        \
  static opt::RegisterHelper g_reg_##backend##_##dynamic##_##origin_op_name( \
    origin_op_name, backend, dynamic, std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value, __VA_ARGS__)
}  // namespace mindspore::opt
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_FACTORY_H_
