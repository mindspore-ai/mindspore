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

#include "include/backend/debug/data_dump/acl_dump_json_writer.h"
#include <string>
#include <fstream>
#include <map>
#include <set>
#include <mutex>
#include <vector>
#include <memory>
#include "nlohmann/json.hpp"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "utils/file_utils.h"
#include "include/common/debug/common.h"

namespace mindspore {
void AclDumpJsonWriter::Parse() {
  auto &dump_parser = DumpJsonParser::GetInstance();
  dump_parser.Parse();
  auto base_dump_path = dump_parser.path();
  MS_LOG(INFO) << "Base dump path is: " << base_dump_path;
  dump_base_path_ = base_dump_path;
  acl_dump_json_path_ = base_dump_path;
  auto dump_mode = dump_parser.input_output();
  MS_LOG(INFO) << "Dump mode is: " << dump_mode;
  switch (dump_mode) {
    case static_cast<uint32_t>(mindspore::DumpJsonParser::JsonInputOutput::DUMP_BOTH):
      dump_mode_ = "all";
      break;
    case static_cast<uint32_t>(mindspore::DumpJsonParser::JsonInputOutput::DUMP_INPUT):
      dump_mode_ = "input";
      break;
    case static_cast<uint32_t>(mindspore::DumpJsonParser::JsonInputOutput::DUMP_OUTPUT):
      dump_mode_ = "output";
      break;
    default:
      dump_mode_ = "all";
      break;
  }
  auto kernels = dump_parser.GetKernelsJson();
  MS_LOG(INFO) << "Dump kernels is as follows: ";
  for (const auto &iter : kernels) {
    MS_LOG(INFO) << iter.dump();
  }
  layer_.clear();
  auto kernel_strings = dump_parser.GetKernelStrs();
  for (auto iter = kernel_strings.begin(); iter != kernel_strings.end(); iter++) {
    auto kernel_str = iter->first;
    if (kernel_str != "") {
      layer_.push_back(kernel_str);
    }
  }
  auto op_debug_mode = dump_parser.op_debug_mode();
  MS_LOG(INFO) << "Op_debug_mode is: " << op_debug_mode;
  switch (op_debug_mode) {
    case 0:
      dump_scene_ = "normal";
      break;
    case 3:
      dump_scene_ = "overflow";
      break;
    case 4:
      dump_scene_ = "lite_exception";
      break;
    default:
      dump_scene_ = "normal";
      break;
  }
}

bool AclDumpJsonWriter::WriteToFile(uint32_t device_id, uint32_t step_id, bool is_init,
                                    nlohmann::json target_kernel_names) {
  nlohmann::json dump_list;

  if (!target_kernel_names.empty()) {
    for (const auto &s : target_kernel_names) {
      layer_.emplace_back(s);
    }
  }
  if (!layer_.empty()) {
    dump_list.push_back({{"layer", layer_}});
  }
  std::string dump_path = dump_base_path_ + "/" + std::to_string(step_id);
  nlohmann::json dump;
  if (dump_scene_ == "lite_exception") {
    dump = {{"dump_scene", "lite_exception"}};
  } else if (dump_scene_ == "overflow") {
    auto real_path = FileUtils::CreateNotExistDirs(dump_path, false);
    if (!real_path.has_value()) {
      MS_LOG(WARNING) << "Fail to create acl dump dir " << real_path.value();
      return false;
    }
    dump = {{"dump_path", dump_path}, {"dump_debug", "on"}};
  } else {
    if (is_init == True) {
      dump = {{"dump_path", dump_base_path_}, {"dump_mode", dump_mode_}, {"dump_step", std::to_string(2147483647)}};
    } else {
      dump = {{"dump_path", dump_path}, {"dump_mode", dump_mode_}};
    }
    if (!dump_list.empty()) {
      dump["dump_list"] = dump_list;
    } else {
      dump["dump_list"] = nlohmann::json::array();
    }
    dump["dump_op_switch"] = "on";
    if (dump_scene_ != "normal") {
      dump["dump_scene"] = dump_scene_;
    }
  }
  nlohmann::json whole_content = {{"dump", dump}};
  std::string json_file_str = whole_content.dump();
  std::string file_name = acl_dump_json_path_ + "/acl_dump_" + std::to_string(device_id) + ".json";
  auto realpath = Common::CreatePrefixPath(file_name);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get realpath failed, path=" << file_name;
    return false;
  }
  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream json_file(realpath.value());
  if (!json_file.is_open()) {
    MS_LOG(EXCEPTION) << "Write file:" << realpath.value() << " open failed."
                      << " Errno:" << errno;
  }
  try {
    json_file << json_file_str;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Write json file: " << file_name << " failed. ";
    json_file.close();
    MS_LOG(EXCEPTION) << "Write dump json failed, error:" << e.what();
  }
  json_file.close();
  MS_LOG(INFO) << "Write to file: " << file_name << " finished.";
  ChangeFileMode(realpath.value(), S_IRUSR);
  return true;
}
}  // namespace mindspore
