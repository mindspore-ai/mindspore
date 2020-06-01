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

#include "debug/trace_base.h"

#include <iostream>
#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <stack>
#include <algorithm>

#include "utils/graph_utils.h"

namespace mindspore {
// namespace to support debug trace infomation
namespace trace {
std::vector<DebugInfoPtr> GetSourceCodeDebugInfoVec(DebugInfoPtr debug_info) {
  std::vector<DebugInfoPtr> debug_with_loc_vec;
  while (debug_info != nullptr) {
    if (debug_info->location() != nullptr) {
      debug_with_loc_vec.push_back(debug_info);
    }
    if (debug_info->trace_info() != nullptr) {
      debug_info = debug_info->trace_info()->debug_info();
    } else {
      break;
    }
  }
  return debug_with_loc_vec;
}

DebugInfoPtr GetSourceCodeDebugInfo(const DebugInfoPtr &info) {
  auto debug_with_loc_vec = GetSourceCodeDebugInfoVec(info);
  if (debug_with_loc_vec.size() > 0) {
    return debug_with_loc_vec[0];
  } else {
    return info;
  }
}

std::string GetDebugInfo(const DebugInfoPtr &info, SourceLineTip tip) {
  if (info == nullptr) {
    return "";
  }
  auto src_info = GetSourceCodeDebugInfo(info);
  if (src_info->location() != nullptr) {
    return src_info->location()->ToString(tip);
  }
  return "";
}

// a trace info identifies a node transform, so we can trace the node transform through
// a link of trace info and debug info
std::string GetInfoWithAction(const std::vector<DebugInfoPtr> &info_vec, SourceLineTip tip) {
  if (info_vec.size() < 1) {
    return "";
  }
  if (info_vec.size() == 1) {
    return info_vec[0]->location()->ToString(tip);
  }
  std::string traced_info = info_vec[0]->location()->ToString(tip);
  for (size_t i = 1; i < info_vec.size(); i++) {
    auto action_name = info_vec[i - 1]->trace_info()->GetActionBetweenNode(info_vec[i]);
    if (action_name == "") {
      break;
    }
    traced_info = traced_info + action_name + info_vec[i]->location()->ToString(tip);
  }
  return traced_info;
}

std::string GetTracedDebugInfo(const DebugInfoPtr &info, SourceLineTip tip) {
  if (info == nullptr) {
    return "";
  }
  auto info_vec = GetSourceCodeDebugInfoVec(info);
  if (info_vec.size() == 0) {
    return "";
  } else if (info_vec.size() == 1) {
    return info_vec[0]->location()->ToString(tip);
  } else if (info_vec.size() > 1) {
    return GetInfoWithAction(info_vec, tip);
  }
  return "";
}

std::string GetDebugInfo(const DebugInfoPtr &info, const std::string &prefix, SourceLineTip tip) {
  std::ostringstream oss;
  if (info == nullptr) {
    return "";
  }

  auto debug_info = GetTracedDebugInfo(info, tip);
  if (tip == kSourceLineTipDiscard) {
    std::replace(debug_info.begin(), debug_info.end(), '\r', '/');
    std::replace(debug_info.begin(), debug_info.end(), '\n', '/');
  }
  oss << prefix << debug_info;
  return oss.str();
}
}  // namespace trace
}  // namespace mindspore
