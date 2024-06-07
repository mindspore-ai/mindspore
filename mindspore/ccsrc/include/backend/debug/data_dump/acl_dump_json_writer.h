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

#ifndef MINDSPORE_ACL_DUMP_JSON_WRITER_H
#define MINDSPORE_ACL_DUMP_JSON_WRITER_H
#include <string>
#include <map>
#include <set>
#include <mutex>
#include <vector>
#include <memory>
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
#include "include/backend/visible.h"

namespace mindspore {
class BACKEND_EXPORT AclDumpJsonWriter {
 public:
  static AclDumpJsonWriter &GetInstance() {
    std::call_once(instance_mutex_, []() {
      if (instance_ == nullptr) {
        instance_ = std::shared_ptr<AclDumpJsonWriter>(new AclDumpJsonWriter);
      }
    });
    return *instance_;
  }
  static void Finalize() { instance_ = nullptr; }

  ~AclDumpJsonWriter() = default;
  // Parse the json file by DumpJsonParser.
  void Parse();
  // Write the parsed feilds to a new json file with acldump format.
  bool WriteToFile(uint32_t device_id = 0, uint32_t step_id = 0, bool is_init = false,
                   nlohmann::json target_kernel_names = nlohmann::json::array());
  std::string GetAclDumpJsonPath() { return acl_dump_json_path_; }

 private:
  AclDumpJsonWriter() = default;
  DISABLE_COPY_AND_ASSIGN(AclDumpJsonWriter)
  inline static std::shared_ptr<AclDumpJsonWriter> instance_ = nullptr;
  inline static std::once_flag instance_mutex_;
  std::string acl_dump_json_path_ = "";
  std::string dump_base_path_ = "";
  std::string dump_mode_ = "all";
  nlohmann::json layer_ = nlohmann::json::array();
  std::string dump_scene_ = "normal";
  std::string dump_debug_ = "off";
};  // class AclDumpJsonWriter
}  // namespace mindspore
#endif  // MINDSPORE_ACL_DUMP_JSON_WRITER_H
