/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_DEBUG_CONF_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_DEBUG_CONF_H_

#include <memory>
#include <map>
#include <string>
#include <nlohmann/json.hpp>

#include "plugin/ascend/res_manager/visible.h"
#include "utils/ms_utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace device {
namespace ascend {
const uint32_t kOpTimeout = 900;
class ASCEND_RES_MANAGER_EXPORT OpDebugConf {
 public:
  ~OpDebugConf() = default;
  OpDebugConf(const OpDebugConf &) = delete;
  OpDebugConf &operator=(const OpDebugConf &) = delete;
  static std::shared_ptr<OpDebugConf> GetInstance();
  uint32_t execute_timeout() const;
  void set_execute_timeout(uint32_t op_timeout);
  std::string debug_option() const;
  void set_debug_option(const std::string &option_value) { debug_option_ = option_value; }
  void set_max_opqueue_num(const std::string &opqueue_num);
  void set_err_msg_mode(const std::string &msg_mode);
  void set_lite_exception_dump(const std::map<std::string, std::string> &);
  bool GenerateAclInitJson(const std::string &file_name, std::string *json_str);
  bool IsExecuteTimeoutConfigured() const { return is_execute_timeout_configured_; }
  bool IsDebugOptionConfigured() const { return !debug_option_.empty(); }

 private:
  nlohmann::json acl_init_json_;
  OpDebugConf();
  static std::shared_ptr<OpDebugConf> inst_context_;
  uint32_t execute_timeout_{kOpTimeout};
  bool is_execute_timeout_configured_{false};
  std::string debug_option_{};
};

ASCEND_RES_MANAGER_EXPORT void RegOpDebugConf(py::module *m);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_DEBUG_CONF_H_
