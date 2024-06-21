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

#include <mutex>
#include <regex>
#include "nlohmann/json.hpp"
#include "debug/data_dump/data_dumper.h"
#include "utils/ms_utils.h"
#include "include/backend/debug/data_dump/acl_dump_json_writer.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "transform/symbol/acl_mdl_symbol.h"
#include "transform/symbol/acl_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace {
bool g_acl_initialized = false;
std::mutex g_acl_init_mutex;

void InitializeAcl() {
  std::lock_guard<std::mutex> lock(g_acl_init_mutex);
  if (g_acl_initialized) {
    return;
  }

  if (CALL_ASCEND_API(aclInit, nullptr) != ACL_ERROR_NONE) {
    MS_LOG(INFO) << "Call aclInit failed, acl data dump function may be unusable.";
  } else {
    MS_LOG(DEBUG) << "Call aclInit successfully";
  }
  g_acl_initialized = true;
}
}  // namespace

namespace mindspore {
namespace datadump {
class AclDataDumper : public DataDumper {
 public:
  void Initialize() override {
    // NOTE: function `aclmdlInitDump` must be called after `aclInit` to take effect, MindSpore never call `aclInit`
    // before, so here call it once
    InitializeAcl();

    if (CALL_ASCEND_API(aclmdlInitDump) != ACL_ERROR_NONE) {
      MS_LOG(INFO) << "Call aclmdlInitDump failed, acl data dump function will be unusable.";
    }
  }
  void EnableDump(uint32_t device_id, uint32_t step_id, bool is_init,
                  const std::vector<std::string> &all_kernel_names) override {
    auto &dump_parser = DumpJsonParser::GetInstance();
    dump_parser.Parse();

    // select target kernel names specified by user config
    nlohmann::json target_kernel_names = nlohmann::json::array();
    auto kernel_regs = dump_parser.GetKernelRegs();
    if (!kernel_regs.empty()) {
      for (auto iter = kernel_regs.begin(); iter != kernel_regs.end(); iter++) {
        auto kernel_reg = iter->second;
        for (const auto &s : all_kernel_names) {
          if (std::regex_search(s, kernel_reg)) {
            target_kernel_names.emplace_back(s);
            dump_parser.MatchKernel(iter->first);
          }
        }
      }
    }

    auto kernels_strs = dump_parser.GetKernelStrs();
    for (auto iter = kernels_strs.begin(); iter != kernels_strs.end(); iter++) {
      for (const auto &s : all_kernel_names) {
        if (iter->first == s) {
          dump_parser.MatchKernel(iter->first);
        }
      }
    }

    if (dump_parser.async_dump_enabled()) {
      auto &acl_json_writer = AclDumpJsonWriter::GetInstance();
      acl_json_writer.Parse();
      acl_json_writer.WriteToFile(device_id, step_id, is_init, target_kernel_names);
      auto acl_dump_file_path = acl_json_writer.GetAclDumpJsonPath();
      std::string json_file_name = acl_dump_file_path + +"/acl_dump_" + std::to_string(device_id) + ".json";
      if (CALL_ASCEND_API(aclmdlSetDump, json_file_name.c_str()) != ACL_ERROR_NONE) {
        MS_LOG(WARNING)
          << "Call aclmdlSetDump failed, acl data dump function will be unusable. Please check whether the config file"
          << json_file_name;
      }
    }
  }

  void Finalize() override {
    if (CALL_ASCEND_API(aclmdlFinalizeDump) != ACL_ERROR_NONE) {
      MS_LOG(WARNING) << "Call aclmdlFinalizeDump failed.";
    }
  }
};

class AclDumpRegister {
 public:
  AclDumpRegister() {
    MS_LOG(INFO) << " Register AclDataDumper for ascend backend\n";
    DataDumperRegister::Instance().RegistDumper(device::DeviceType::kAscend, std::make_shared<AclDataDumper>());
  }

  ~AclDumpRegister() = default;
} g_acl_dump_register;

}  // namespace datadump
}  // namespace mindspore
