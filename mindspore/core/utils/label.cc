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

#include <vector>
#include "utils/info.h"

namespace {
using mindspore::DebugInfoPtr;
using mindspore::TraceInfoPtr;
using mindspore::trace::TraceLabelType;

static const TraceLabelType global_trace_type = (mindspore::common::GetEnv("MS_DEV_TRACE_LABEL_WITH_UNIQUE_ID") == "1")
                                                  ? TraceLabelType::kWithUniqueId
                                                  : TraceLabelType::kShortSymbol;

TraceLabelType GetCurrentTraceLabelType() {
  if (mindspore::common::GetEnv("MS_DEV_TRACE_LABEL_WITH_UNIQUE_ID") == "1") {
    return TraceLabelType::kWithUniqueId;
  }
  return TraceLabelType::kShortSymbol;
}

std::string CombineUniqueID(const DebugInfoPtr &debug_info) {
  auto root_info = debug_info;
  std::string label = "";
  while (root_info != nullptr) {
    if (!root_info->name().empty()) {
      label = label + root_info->name();
    } else {
      // The symbol 'U' is for identification of number
      label = label + "U" + std::to_string(root_info->unique_id());
    }

    if (root_info->trace_info() != nullptr) {
      label = label + "_" + root_info->trace_info()->full_name() + "_";
      root_info = root_info->trace_info()->debug_info();
    } else {
      root_info = nullptr;
    }
  }
  return label;
}

// Get trace with unique id chain
std::string LabelStringUnique(const DebugInfoPtr &debug_info) { return CombineUniqueID(debug_info); }

struct NameWithTrace {
  std::string root_name;
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

NameWithTrace CollectTraceInfos(const DebugInfoPtr &debug_info, TraceLabelType trace_label) {
  NameWithTrace name_and_traces;
  // Find debug info after Resolve/ExpandJ/GenMetaFuncGraph/GenerateVarArg/GenerateKwArg, it is a new node.
  MS_EXCEPTION_IF_NULL(debug_info);
  auto root_info = debug_info;
  while (root_info != nullptr) {
    if (root_info->trace_info() == nullptr) {
      break;
    }
    if (root_info->trace_info()->isa<mindspore::TraceParse>() ||
        root_info->trace_info()->isa<mindspore::TraceResolve>() ||
        root_info->trace_info()->isa<mindspore::TraceExpandJ>() ||
        root_info->trace_info()->isa<mindspore::TraceGenMetaFuncGraph>() ||
        root_info->trace_info()->isa<mindspore::TraceGenerateVarArg>() ||
        root_info->trace_info()->isa<mindspore::TraceGenerateKwArg>()) {
      break;
    }
    name_and_traces.trace_labels.push_back(GetTraceName(root_info->trace_info(), trace_label));
    root_info = root_info->trace_info()->debug_info();
  }

  if (!root_info->name().empty()) {
    name_and_traces.root_name = root_info->name();
  } else {
    name_and_traces.root_name = root_info->debug_name();
  }
  return name_and_traces;
}

std::string CombineTraceInfos(const std::string &root_name, const std::vector<std::string> &trace_labels) {
  std::stringstream ss_labels;
  for (size_t i = 0; i < trace_labels.size(); ++i) {
    size_t start = i;
    auto &start_label = trace_labels[start];
    if (start_label.empty()) {
      continue;
    }
    // Combine the same continuous symbols. For example, AAA --> 3A
    while (i + 1 < trace_labels.size() && trace_labels[i + 1] == start_label) {
      ++i;
    }
    if (start == i) {
      ss_labels << start_label;
    } else {
      ss_labels << std::to_string(i - start + 1) << start_label;
    }
  }
  ss_labels << root_name;
  return ss_labels.str();
}

// Get the label name of the node debug info
std::string LabelString(const DebugInfoPtr &debug_info, TraceLabelType trace_label) {
  NameWithTrace name_and_traces = CollectTraceInfos(debug_info, trace_label);
  return CombineTraceInfos(name_and_traces.root_name, name_and_traces.trace_labels);
}
}  // namespace

namespace mindspore {
namespace trace {
TraceLabelType GetGlobalTraceLabelType() { return global_trace_type; }

std::string Label(const DebugInfoPtr &debug_info, TraceLabelType trace_label) {
  if ((GetGlobalTraceLabelType() == TraceLabelType::kWithUniqueId) ||
      (GetCurrentTraceLabelType() == TraceLabelType::kWithUniqueId)) {
    return LabelStringUnique(debug_info);
  }
  return LabelString(debug_info, trace_label);
}
}  // namespace trace
}  // namespace mindspore
