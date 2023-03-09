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
#include <utility>
#include <vector>
#include <memory>
#include <map>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "include/backend/visible.h"

namespace mindspore::opt {
class BACKEND_EXPORT OpAdaptationInfo {
 public:
  explicit OpAdaptationInfo(const std::string &me_op_name, std::string device_name, bool flag)
      : me_op_name_(me_op_name),
        backend_op_name_(me_op_name),
        target_op_name_(me_op_name),
        device_name_(std::move(device_name)),
        flag_(flag) {}

  OpAdaptationInfo &operator=(const OpAdaptationInfo &op_adaptation_info);
  virtual ~OpAdaptationInfo() = default;

  OpAdaptationInfo &set_backend_op_name(const std::string &default_op_name);
  OpAdaptationInfo &set_target_op_name(const std::string &target_op_name);
  OpAdaptationInfo &set_pre_check_func(std::function<bool(CNodePtr)> pre_check_func);
  OpAdaptationInfo &set_need_tbe_check_supported(bool need_tbe_check_supported);
  OpAdaptationInfo &set_input_attr_info(size_t input_index, std::string attr_data_type = "");

  const std::string &me_op_name() const { return me_op_name_; }
  const std::string &backend_op_name() const { return backend_op_name_; }
  const std::string &target_op_name() const { return target_op_name_; }
  const std::function<bool(CNodePtr)> &pre_check_func() const { return pre_check_func_; }
  bool need_tbe_check_supported() const { return need_tbe_check_supported_; }
  const std::map<size_t, std::string> &input_attr_map() const { return input_attr_map_; }
  const std::string &device_name() const { return device_name_; }
  bool flag() const { return flag_; }

 private:
  std::string me_op_name_;
  std::string backend_op_name_;
  std::string target_op_name_;
  std::function<bool(CNodePtr)> pre_check_func_{nullptr};
  bool need_tbe_check_supported_{false};
  std::map<size_t, std::string> input_attr_map_;
  std::string device_name_;
  bool flag_{false};
};

class BACKEND_EXPORT OpAdaptationInfoRegister {
 public:
  static OpAdaptationInfoRegister &GetInstance();
  void RegOpAdaptationInfo(OpAdaptationInfo *reg_info);
  [[nodiscard]] OpAdaptationInfo *GetOpAdaptationInfo(const std::string &me_op_name, const std::string &device_name,
                                                      bool flag) const;

 private:
  OpAdaptationInfoRegister() = default;
  ~OpAdaptationInfoRegister() = default;
  DISABLE_COPY_AND_ASSIGN(OpAdaptationInfoRegister)

  static std::string GenerateKey(const std::string &me_op_name, const std::string &device_name, bool flag);
  // key: (op_name + device_name + flag), value: <OpAdaptationInfo *>
  static std::map<std::string, OpAdaptationInfo *> &GetOpInfoMap();
};

class BACKEND_EXPORT RegisterHelper {
 public:
  RegisterHelper(const std::string &me_op_name, const std::string &device_name, bool flag, int len, ...);
  RegisterHelper(const OpAdaptationInfo &op_adaptation_info);
  ~RegisterHelper() = default;

 private:
  std::shared_ptr<OpAdaptationInfo> op_adaptation_info_{nullptr};
};

#define REG_OP_ADAPTATION_INFO(me_op_name, device_name, flag)                                      \
  static opt::RegisterHelper g_reg_##device_name##_##flag##_##me_op_name __attribute__((unused)) = \
    opt::OpAdaptationInfo(me_op_name, device_name, flag)

#define RER_CONST_TO_ATTR_LIST(me_op_name, device_name, flag, ...)        \
  static opt::RegisterHelper g_reg_##device_name##_##flag##_##me_op_name( \
    me_op_name, device_name, flag, std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value, __VA_ARGS__)
}  // namespace mindspore::opt
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_FACTORY_H_
