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

#include "cxx_api/model/aoe/auto_tune_process.h"
#include <cstdio>
#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <map>

namespace mindspore {
namespace {
constexpr int kBuffSize = 1024;
const std::map<std::string, std::string> kTuneModeMap = {{"1", "subgraph tuning"}, {"2", "operator tuning"}};
}  // namespace

static std::string SaveAirGraphToFile(const std::shared_ptr<AclModelOptions> &options,
                                      const transform::DfGraphPtr &graph) {
  std::string air_path = options->GetOmFilePath();
  if (air_path.empty()) {
    air_path = "/tmp";
    MS_LOG(INFO) << "Air path of options is empty, set default path: /tmp";
  }
  auto dir_pos = air_path.find_last_of('/');
  if (dir_pos != std::string::npos) {
    air_path = "/tmp" + air_path.substr(dir_pos) + ".air";
  } else {
    air_path = "/tmp/" + air_path + ".air";
  }
  MS_LOG(INFO) << "Air graph file path: " << air_path;
  auto ret = graph->SaveToFile(air_path.c_str());
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Failed to save air graph.";
    return "";
  }
  return air_path;
}

static std::vector<std::string> GetAoeMode(const std::shared_ptr<AclModelOptions> &options) {
  std::string aoe_mode = options->GetAoeMode();
  std::vector<std::string> tune_mode;
  // 1: subgraph tuning; 2: operator tuning
  if (aoe_mode.find("subgraph tuning") != std::string::npos) {
    tune_mode.emplace_back("1");
  }
  if (aoe_mode.find("operator tuning") != std::string::npos) {
    tune_mode.emplace_back("2");
  }
  if (tune_mode.empty()) {
    MS_LOG(ERROR) << "Aoe mode " << aoe_mode << " are invalid "
                  << "; It should be in 'subgraph tuning, operator tuning'";
    return tune_mode;
  }
  return tune_mode;
}

static Status ExecuteAoe(const std::shared_ptr<AclModelOptions> &options, const transform::DfGraphPtr &graph,
                         const std::string &air_path) {
  MS_LOG(INFO) << "Start to aoe.";
  std::string aoe_path = "aoe";  //  real path is already set in env PATH
  auto aoe_modes = GetAoeMode(options);
  std::map<std::string, std::string> init_options;
  std::map<std::string, std::string> build_options;
  std::tie(init_options, build_options) = options->GenAclOptions();
  std::string dynamic_option;
  if (build_options.find(ge::ir_option::DYNAMIC_BATCH_SIZE) != build_options.end()) {
    dynamic_option = " --dynamic_batch_size=\"" + build_options[ge::ir_option::DYNAMIC_BATCH_SIZE] + "\"";
  } else if (build_options.find(ge::ir_option::DYNAMIC_IMAGE_SIZE) != build_options.end()) {
    dynamic_option = " --dynamic_image_size=\"" + build_options[ge::ir_option::DYNAMIC_IMAGE_SIZE] + "\"";
  }
  std::string input_shape;
  if (build_options.find(ge::ir_option::INPUT_SHAPE) != build_options.end()) {
    input_shape = " --input_shape=\"" + build_options[ge::ir_option::INPUT_SHAPE] + "\"";
  }
  try {
    for (auto &mode : aoe_modes) {
      std::cout << "Start to " << kTuneModeMap.at(mode) << std::endl;
      std::string cmd =
        aoe_path + " --framework=1" + " --model=" + air_path + " --job_type=" + mode + dynamic_option + input_shape;
      MS_LOG(DEBUG) << "Aoe cmd is " << cmd;
      auto fp = popen(cmd.c_str(), "r");
      std::string result;
      if (fp != nullptr) {
        char buf[kBuffSize] = {0};
        while (fgets(buf, kBuffSize, fp) != nullptr) {
          result += buf;
        }
        (void)pclose(fp);
      }
      if (result.find("Aoe process finished") == std::string::npos) {
        MS_LOG(ERROR) << "Aoe process failed, mode= " << kTuneModeMap.at(mode);
        return kMCFailed;
      }
      MS_LOG(INFO) << result;
      std::cout << result << std::endl;
    }
    return kSuccess;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Execute aoe failed: " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "Execute aoe failed.";
  }
  return kMCFailed;
}

Status AutoTuneProcess::AoeOfflineTurningGraph(const std::weak_ptr<AclModelOptions> &options,
                                               const transform::DfGraphPtr &graph) {
  auto option_ptr = options.lock();
  if (option_ptr == nullptr) {
    MS_LOG(ERROR) << "Option ptr is nullptr.";
    return kMCFailed;
  }
  if (option_ptr->GetAoeMode().empty()) {
    MS_LOG(DEBUG) << "Aoe mode is empty, no need to enable aoe.";
    return kSuccess;
  }
  auto air_path = SaveAirGraphToFile(option_ptr, graph);
  if (air_path.empty()) {
    MS_LOG(ERROR) << "Save air graph to file failed.";
    return kMCFailed;
  }
  if (ExecuteAoe(option_ptr, graph, air_path) != kSuccess) {
    MS_LOG(ERROR) << "Execute aoe failed, air path=" << air_path;
    return kMCFailed;
  }
  if (remove(air_path.c_str()) != 0) {
    MS_LOG(ERROR) << "Remove air file failed, file path= " << air_path;
    return kMCFailed;
  }
  return kSuccess;
}
}  // namespace mindspore
