/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "utils/info.h"
#include <fstream>
#include <sstream>
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "ir/func_graph.h"
#include "utils/convert_utils_base.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace {
/// \brief Trace context stack for current thread.
thread_local std::vector<TraceContext> trace_context_stack_;

/// \brief Record a debug info for print.
thread_local DebugInfoPtr record_debug_info_ = nullptr;

/// \brief A flag to decide whether record a debug info or not.
thread_local bool record_debug_info_flag_ = false;
}  // namespace

void ClearThreadLocal() {
  trace_context_stack_.clear();
  record_debug_info_.reset();
}

std::string HighLightLine(const std::string &line, int col_begin, int col_end, SourceLineTip tip) {
  std::string temp_line = line;
  if (col_begin < col_end && col_begin != -1 && col_end <= SizeToLong(temp_line.length()) &&
      tip != kSourceLineTipDiscard) {
    std::string start = temp_line.substr(0, LongToSize(col_begin));
    std::string trimmed = temp_line.substr(LongToSize(col_begin), LongToSize(col_end - col_begin));
    std::string end = temp_line.substr(LongToSize(col_end), LongToSize(SizeToLong(temp_line.length()) - col_end));
    std::stringstream oss;
    std::stringstream tip_ss;
    std::string start_spaces(start.length(), ' ');
    if (tip == kSourceLineTipInLine) {
      temp_line = start + "<" + trimmed + ">" + end;
    } else if (tip == kSourceLineTipNextLine) {
      tip_ss << start_spaces << "^";
    }
    oss << temp_line << "\n" << tip_ss.str();
    return oss.str();
  }
  return temp_line;
}

// Generate debug information for the location node .
// print the file name, line no and column no, and part of the content
std::string Location::ToString(SourceLineTip tip) const {
  std::stringstream debug_info_ss;
  debug_info_ss << "In file " << file_name_ << ":" << line_ << std::endl;
  if (line_ <= 0) {
    return debug_info_ss.str();
  }
  auto path = FileUtils::GetRealPath(file_name_.c_str());
  if (!path.has_value()) {
    return debug_info_ss.str();
  }
  std::ifstream file(path.value());
  if (!file.is_open()) {
    return debug_info_ss.str();
  }

  int line_num = 0;
  std::string line;
  (void)getline(file, line);
  while (line_num != line_ - 1) {
    (void)getline(file, line);
    line_num++;
  }
  file.close();

  debug_info_ss << HighLightLine(line, column_, column_end_, tip) << std::endl;
  return debug_info_ss.str();
}

bool Location::operator<(const Location &other) const {
  auto ret = file_name_.compare(other.file_name());
  if (ret != 0) {
    return ret < 0;
  }
  return line_ < other.line();
}

int64_t DebugInfo::get_id() const {
  // cppcheck-suppress variableScope
  static int64_t current_id = 1;
  if (id_ == 0) {
    id_ = current_id++;
  }
  return id_;
}

int64_t DebugInfo::unique_id_through_copy() const {
  auto info = trace_info();
  if (info != nullptr) {
    if (info->isa<TraceCopy>() && info->debug_info() != nullptr) {
      return info->debug_info()->unique_id_through_copy();
    }
  }
  return unique_id();
}

std::string NodeDebugInfo::debug_name() {
  if (!name_.empty()) {
    return name_;
  }
  std::string prefix = "";
  if (node_.lock() != nullptr) {
    std::ostringstream oss;
    oss << "[" << node_.lock()->type_name() << "]";
    prefix = oss.str();
  }
  name_ = prefix + std::to_string(get_id());
  return name_;
}

std::string GraphDebugInfo::debug_name() {
  if (name_.empty()) {
    name_ = std::to_string(get_id());
  }
  return name_;
}

LocationPtr GraphDebugInfo::location() const {
  // Function may have decorator which is included in its location.
  auto loc = DebugInfo::location();
  if (deco_loc_ != nullptr && loc != nullptr) {
    auto loc_line = loc->line() + ((deco_loc_->line_end() - deco_loc_->line()) + 1);
    return std::make_shared<Location>(loc->file_name(), loc_line, loc->line_end(), loc->column(), loc->column_end(),
                                      loc->expr_src());
  }
  return loc;
}

void GraphDebugInfo::set_deco_location(const LocationPtr &deco_list_loc) { deco_loc_ = deco_list_loc; }

TraceContextPtr TraceManager::CurrentContextInfo() {
  if (!trace_context_stack_.empty()) {
    return &trace_context_stack_.back();
  }
  return nullptr;
}

void TraceManager::DebugTrace(const std::string &func_name, const LocationPtr &location) {
  MS_EXCEPTION_IF_NULL(location);
  (void)trace_context_stack_.emplace_back(location, func_name);
}

void TraceManager::DebugTrace(const LocationPtr &location) {
  MS_EXCEPTION_IF_NULL(location);
  (void)trace_context_stack_.emplace_back(location);
  if (record_debug_info_flag_) {
    record_debug_info_ = std::make_shared<DebugInfo>(location);
  }
}

void TraceManager::DebugTrace(const TraceInfoPtr &trace_info) {
  MS_EXCEPTION_IF_NULL(trace_info);
  auto &debug_info = trace_info->debug_info();
  MS_EXCEPTION_IF_NULL(debug_info);
  (void)trace_context_stack_.emplace_back(trace_info);
  if (record_debug_info_flag_) {
    record_debug_info_ = debug_info;
  }
}

void TraceManager::DebugTrace(const DebugInfoPtr &debug_info, const TraceInfoPtr &trace_info) {
  MS_EXCEPTION_IF_NULL(debug_info);
  MS_EXCEPTION_IF_NULL(trace_info);
  auto cloned_info = trace_info->clone();
  cloned_info->set_debug_info(debug_info);
  (void)trace_context_stack_.emplace_back(cloned_info);
}

void TraceManager::EndTrace() noexcept {
  trace_context_stack_.pop_back();
  ClearParseOrResolveDebugInfo();
}

DebugInfoPtr TraceManager::record_debug_info() { return record_debug_info_; }

void TraceManager::ClearParseOrResolveDebugInfo() { record_debug_info_ = nullptr; }

void TraceManager::CloseRecordDebugInfoFlag() { record_debug_info_flag_ = false; }

void TraceManager::OpenRecordDebugInfoFlag() { record_debug_info_flag_ = true; }

bool TraceManager::record_debug_info_flag() { return record_debug_info_flag_; }

LocationPtr GetFirstLocation(const DebugInfoPtr &debug_info) {
  auto tmp = debug_info;
  while (tmp != nullptr) {
    if (tmp->location() != nullptr) {
      return tmp->location();
    }
    if (tmp->trace_info() != nullptr) {
      tmp = tmp->trace_info()->debug_info();
    } else {
      break;
    }
  }
  return nullptr;
}

bool DebugInfoCompare::operator()(const DebugInfoPtr &left, const DebugInfoPtr &right) const {
  MS_EXCEPTION_IF_NULL(left);
  MS_EXCEPTION_IF_NULL(right);
  if (left == right) {
    return false;
  }
  auto left_loc = GetFirstLocation(left);
  auto right_loc = GetFirstLocation(right);
  if (left_loc == nullptr || right_loc == nullptr) {
    return left < right;
  }
  if (left_loc == right_loc) {
    return false;
  }
  return *left_loc < *right_loc;
}

void UpdateDebugInfo(const FuncGraphPtr &func_graph, const ScopePtr &scope, const DebugInfoPtr &debug_info) {
  if (func_graph == nullptr || scope == nullptr || debug_info == nullptr) {
    return;
  }
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  TraceGuard guard(std::make_shared<TraceGenMetaFuncGraph>(debug_info));
  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    node->set_scope(std::make_shared<Scope>(scope->name()));
    node->set_debug_info(std::make_shared<NodeDebugInfo>());
  }
}
}  // namespace mindspore
