/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_BASE_H_
#define MINDSPORE_CCSRC_PIPELINE_BASE_H_

#include <mutex>
#include <memory>
#include <string>
#include <sstream>

#include "ir/anf.h"
#include "pipeline/resource.h"
#include "utils/context/ms_context.h"

namespace mindspore {
namespace pipeline {

struct ExecutorInfo {
  FuncGraphPtr func_graph;
  ResourcePtr resource;
  std::size_t arg_list_size;
};

using ExecutorInfoPtr = std::shared_ptr<ExecutorInfo>;

inline std::string GetPhasePrefix(const std::string& phase) {
  auto pos = phase.find('.');
  if (pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "phase has no . for prefix" << phase;
  }
  return phase.substr(0, pos);
}

inline std::string GetFilePathName(const std::string& file_name) {
  std::ostringstream oss;
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(EXCEPTION) << "ms_context is nullptr";
  }
  auto save_graphs_path = ms_context->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  oss << save_graphs_path << "/" << file_name;
  return oss.str();
}
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_BASE_H_
