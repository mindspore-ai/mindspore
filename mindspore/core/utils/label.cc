/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "utils/label.h"

#include "utils/info.h"

namespace mindspore {
namespace label_manage {
static const TraceLabelType global_trace_type = (common::GetEnv("MS_DEV_TRACE_LABEL_WITH_UNIQUE_ID") == "1")
                                                  ? TraceLabelType::kWithUniqueId
                                                  : TraceLabelType::kShortSymbol;
TraceLabelType GetGlobalTraceLabelType() { return global_trace_type; }
TraceLabelType GetCurrentTraceLabelType() {
  if (common::GetEnv("MS_DEV_TRACE_LABEL_WITH_UNIQUE_ID") == "1") {
    return TraceLabelType::kWithUniqueId;
  }
  return TraceLabelType::kShortSymbol;
}

struct NameWithTrace {
  std::string name;
  std::vector<std::string> trace_labels;
};
static std::string GetTraceName(const TraceInfoPtr &trace_info, TraceLabelType trace_label) {
  switch (trace_label) {
    case TraceLabelType::kShortSymbol:
      return trace_info->symbol();
    case TraceLabelType::kFullName:
      return "_" + trace_info->full_name() + "_";
    default:
      return "";
  }
}

NameWithTrace RootName(const DebugInfoPtr &debug_info, TraceLabelType trace_label) {
  NameWithTrace trace_name;
  // find debug info after Resolve/ExpandJ/GenMetaFuncGraph, it is a new node
  MS_EXCEPTION_IF_NULL(debug_info);
  auto temp_info = debug_info;
  while (temp_info != nullptr) {
    if (temp_info->trace_info() != nullptr) {
      if (temp_info->trace_info()->isa<TraceResolve>() || temp_info->trace_info()->isa<TraceExpandJ>() ||
          temp_info->trace_info()->isa<TraceGenMetaFuncGraph>() ||
          temp_info->trace_info()->isa<TraceGenerateVarArg>() || temp_info->trace_info()->isa<TraceGenerateKwArg>()) {
        break;
      }
      trace_name.trace_labels.push_back(GetTraceName(temp_info->trace_info(), trace_label));
      temp_info = temp_info->trace_info()->debug_info();
    } else {
      break;
    }
  }
  if (!temp_info->name().empty()) {
    trace_name.name = temp_info->name();
  } else {
    trace_name.name = temp_info->debug_name();
  }
  return trace_name;
}

std::string CombineTraceTypes(const std::string &root_name, const std::vector<std::string> &trace_labels) {
  std::string tags = "";
  for (auto &itr : trace_labels) {
    std::string symbol = itr;
    tags = tags + symbol;
  }
  return tags + root_name;
}

// get the label name of the node debug info
std::string LabelString(const DebugInfoPtr &debug_info, TraceLabelType trace_label) {
  NameWithTrace trace_name = RootName(debug_info, trace_label);
  return CombineTraceTypes(trace_name.name, trace_name.trace_labels);
}

std::string CombineUniqueID(const DebugInfoPtr &debug_info) {
  auto temp_info = debug_info;
  std::string label = "";
  while (temp_info != nullptr) {
    if (!temp_info->name().empty()) {
      label = label + temp_info->name();
    } else {
      // the symbol 'U' is for identification of number
      label = label + "U" + std::to_string(temp_info->unique_id());
    }

    if (temp_info->trace_info() != nullptr) {
      label = label + "_" + temp_info->trace_info()->full_name() + "_";
      temp_info = temp_info->trace_info()->debug_info();
    } else {
      temp_info = nullptr;
    }
  }
  return label;
}

// get trace with unique id chain
std::string LabelStringUnique(const DebugInfoPtr &debug_info) { return CombineUniqueID(debug_info); }

std::string Label(const DebugInfoPtr &debug_info, TraceLabelType trace_label) {
  if ((GetGlobalTraceLabelType() == TraceLabelType::kWithUniqueId) ||
      (GetCurrentTraceLabelType() == TraceLabelType::kWithUniqueId)) {
    return LabelStringUnique(debug_info);
  }
  return LabelString(debug_info, trace_label);
}
}  // namespace label_manage
}  // namespace mindspore
