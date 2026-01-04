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
#include "plugin/ascend/res_manager/device_context_conf/op_debug_conf.h"
#include <map>
#include <fstream>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include "include/utils/common.h"
#include "pybind11/stl.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr auto err_msg_mode = "err_msg_mode";
constexpr auto default_err_msg_mode = "1";
constexpr auto max_opqueue_num = "max_opqueue_num";
constexpr auto dump = "dump";
constexpr auto dump_scene = "dump_scene";
constexpr auto lite_exception = "lite_exception";
constexpr auto lite_exception_disable = "lite_exception:disable";
}  // namespace

std::shared_ptr<OpDebugConf> OpDebugConf::inst_context_ = nullptr;

std::shared_ptr<OpDebugConf> OpDebugConf::GetInstance() {
  static std::once_flag inst_context_init_flag_ = {};
  std::call_once(inst_context_init_flag_, [&]() {
    if (inst_context_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore OpDebugConf";
      inst_context_ = std::shared_ptr<OpDebugConf>(new OpDebugConf());
    }
  });
  MS_EXCEPTION_IF_NULL(inst_context_);
  return inst_context_;
}

OpDebugConf::OpDebugConf() {
  acl_init_json_[err_msg_mode] = default_err_msg_mode;
  acl_init_json_[dump][dump_scene] = lite_exception;
}

void OpDebugConf::set_execute_timeout(uint32_t op_timeout) {
  is_execute_timeout_configured_ = true;
  execute_timeout_ = op_timeout;
}

void OpDebugConf::set_lite_exception_dump(const std::map<std::string, std::string> &dump_config) {
  auto it = dump_config.find(dump_scene);
  if (it == dump_config.end()) {
    return;
  }
  if (it->second == lite_exception_disable) {
    acl_init_json_.erase(dump);
  }
}

uint32_t OpDebugConf::execute_timeout() const {
  if (is_execute_timeout_configured_) {
    return execute_timeout_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  uint32_t execute_timeout = ms_context->get_param<uint32_t>(MS_CTX_OP_TIMEOUT);
  return execute_timeout;
}

std::string OpDebugConf::debug_option() const {
  if (!debug_option_.empty()) {
    return debug_option_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto debug_option = ms_context->get_param<std::string>(MS_CTX_OP_DEBUG_OPTION);
  return debug_option;
}

bool OpDebugConf::GenerateAclInitJson(const std::string &file_path, std::string *json_str) {
  // write to file
  *json_str = acl_init_json_.dump();
  std::ofstream json_file(file_path);
  if (!json_file.is_open()) {
    MS_LOG(WARNING) << "Open file [" << file_path << "] failed!";
    return false;
  }
  json_file << *json_str;
  json_file.close();
  return true;
}

void OpDebugConf::set_max_opqueue_num(const std::string &opqueue_num) {
  if (acl_init_json_[max_opqueue_num] == opqueue_num) {
    return;
  }
  acl_init_json_[max_opqueue_num] = opqueue_num;
}

void OpDebugConf::set_err_msg_mode(const std::string &msg_mode) {
  if (acl_init_json_[err_msg_mode] == msg_mode) {
    return;
  }
  acl_init_json_[err_msg_mode] = msg_mode;
}

void RegOpDebugConf(py::module *m) {
  (void)py::class_<OpDebugConf, std::shared_ptr<OpDebugConf>>(*m, "AscendOpDebugConf")
    .def_static("get_instance", &OpDebugConf::GetInstance, "Get OpDebugConf instance.")
    .def("set_execute_timeout", &OpDebugConf::set_execute_timeout, "Set Execute Timeout.")
    .def("execute_timeout", &OpDebugConf::execute_timeout, "Get Execute Timeout.")
    .def("set_debug_option", &OpDebugConf::set_debug_option, "Set Debug Option.")
    .def("debug_option", &OpDebugConf::debug_option, "Get Debug Option.")
    .def("set_lite_exception_dump", &OpDebugConf::set_lite_exception_dump, "Set lite exception dump")
    .def("is_execute_timeout_configured", &OpDebugConf::IsExecuteTimeoutConfigured, "Is Execute Timeout Configured.")
    .def("is_debug_option_configured", &OpDebugConf::IsDebugOptionConfigured, "Is Debug Option Configured.")
    .def("set_max_opqueue_num", &OpDebugConf::set_max_opqueue_num, "set_max_opqueue_num")
    .def("set_err_msg_mode", &OpDebugConf::set_err_msg_mode, "set_err_msg_mode");
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
