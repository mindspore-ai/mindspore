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
#include "utils/trace_base.h"

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
  auto final_unique_id = through_copy_unique_id_ == -1 ? unique_id_ : through_copy_unique_id_;
  if (info != nullptr) {
    if (info->isa<TraceCopy>() && info->debug_info() != nullptr) {
      final_unique_id = info->debug_info()->unique_id_through_copy();
    }
  }
  const_cast<DebugInfo *>(this)->through_copy_unique_id_ = final_unique_id;
  return final_unique_id;
}

DebugInfoPtr DebugInfo::Copy() {
  auto new_debug_info = std::make_shared<NodeDebugInfo>();
  new_debug_info->location_ = location_;
  new_debug_info->trace_info_ = trace_info_;
  new_debug_info->id_ = id_;
  new_debug_info->name_ = name_;
  new_debug_info->unique_id_ = unique_id_;
  new_debug_info->through_copy_unique_id_ = through_copy_unique_id_;
  new_debug_info->inlined_ = inlined_;
  return new_debug_info;
}

HashSet<DebugInfoPtr> GetDebugInfos(const DebugInfoPtr &debug_info) {
  HashSet<DebugInfoPtr> debug_infos;
  auto cur_debug_info = debug_info;
  while (cur_debug_info != nullptr) {
    (void)debug_infos.insert(cur_debug_info);
    if (cur_debug_info->trace_info() == nullptr) {
      break;
    }
    cur_debug_info = cur_debug_info->trace_info()->debug_info();
  }
  return debug_infos;
}

namespace {
std::pair<DebugInfoPtr, DebugInfoPtr> GetConcatDebugInfo(const DebugInfoPtr &start, const DebugInfoPtr &end,
                                                         const DebugInfoPtr &call) {
  auto call_debug_infos = GetDebugInfos(call);
  auto cur = start;
  DebugInfoPtr prev = nullptr;
  while (cur != end) {
    // Repeat debug info exist, use repeat and prev as concat debug infos.If prev is nullptr, no need concat.
    if (call_debug_infos.find(cur) != call_debug_infos.cend()) {
      return {prev, cur};
    }
    prev = cur;
    cur = cur->trace_info()->debug_info();
  }
  // No repeat debug info exist. Need find the first debug info which has location.
  return {end, call};
}

DebugInfoPtr GetFirstNotInlineDebugInfo(const DebugInfoPtr &info) {
  auto cur_debug_info = info;
  while (cur_debug_info != nullptr) {
    if (!cur_debug_info->inlined() && cur_debug_info->location() != nullptr) {
      break;
    }
    if (cur_debug_info->trace_info() == nullptr) {
      MS_LOG(DEBUG) << "Get null trace info, but first not inline debug info not found.";
      return nullptr;
    }
    cur_debug_info = cur_debug_info->trace_info()->debug_info();
  }
  if (cur_debug_info == nullptr) {
    MS_LOG(DEBUG) << "All debug infos are inlined.";
  }
  return cur_debug_info;
}

std::vector<DebugInfoPtr> CopyDebugInfoLink(const DebugInfoPtr &start, const DebugInfoPtr &end) {
  auto cur_debug_info = start;
  std::vector<DebugInfoPtr> copy_debug_infos;
  (void)start->unique_id_through_copy();
  while (True) {
    MS_EXCEPTION_IF_NULL(cur_debug_info);
    copy_debug_infos.push_back(cur_debug_info->Copy());
    if (cur_debug_info == end) {
      break;
    }

    if (cur_debug_info->trace_info() == nullptr) {
      MS_LOG(EXCEPTION) << "Reach nullptr but not found end.";
    }
    cur_debug_info = cur_debug_info->trace_info()->debug_info();
  }
  for (size_t index = 0; index + 1 < copy_debug_infos.size(); ++index) {
    auto new_trace_info = copy_debug_infos[index]->trace_info()->clone();
    new_trace_info->set_debug_info(copy_debug_infos[index + 1]);
    copy_debug_infos[index]->set_trace_info(new_trace_info);
  }
  return copy_debug_infos;
}

DebugInfoPtr GetFirstHasLocationDebugInfo(const DebugInfoPtr &debug_info) {
  auto first_debug_info = debug_info;
  if (debug_info == nullptr) {
    return nullptr;
  }
  while (first_debug_info->location() == nullptr) {
    // The original debug info is null.
    if (first_debug_info->trace_info() == nullptr) {
      return nullptr;
    }
    first_debug_info = first_debug_info->trace_info()->debug_info();
    if (debug_info == nullptr) {
      return nullptr;
    }
  }
  return first_debug_info;
}
}  // namespace

DebugInfoPtr DebugInfo::UpdateInlineCNodeDebugInfo(const DebugInfoPtr &call_debug_info,
                                                   const DebugInfoPtr &debug_info) {
  static auto enable_fix_code_line = (common::GetEnv("MS_DEV_ENABLE_FIX_CODE_LINE") != "0");
  if (!enable_fix_code_line) {
    return debug_info;
  }
  MS_EXCEPTION_IF_NULL(debug_info);
  MS_LOG(DEBUG) << "Origin source lines:\n" << mindspore::ToString(trace::GetSourceLineList(debug_info));
  MS_LOG(DEBUG) << "call_debug_info source lines:\n" << mindspore::ToString(trace::GetSourceLineList(call_debug_info));
  auto call_first_location_debug_info = GetFirstHasLocationDebugInfo(call_debug_info);
  // If call has no location info , do nothing.
  if (call_first_location_debug_info == nullptr) {
    return debug_info;
  }
  auto first_not_inlined_info = GetFirstNotInlineDebugInfo(debug_info);
  if (first_not_inlined_info == nullptr) {
    return debug_info;
  }
  MS_LOG(DEBUG) << "first_not_inlined_info debug info source lines:\n"
                << mindspore::ToString(trace::GetSourceLineList(first_not_inlined_info));
  auto [concat_pre, concat] = GetConcatDebugInfo(debug_info, first_not_inlined_info, call_first_location_debug_info);
  MS_LOG(DEBUG) << "concat_pre debug info source lines:\n" << mindspore::ToString(trace::GetSourceLineList(concat_pre));
  MS_LOG(DEBUG) << "concat debug info source lines:\n" << mindspore::ToString(trace::GetSourceLineList(concat));
  if (concat_pre == nullptr) {
    MS_LOG(DEBUG) << "Origin debug info exist in call debug info, no need concat.";
    return debug_info;
  }
  auto copy_debug_infos = CopyDebugInfoLink(debug_info, concat_pre);
  std::for_each(copy_debug_infos.cbegin(), copy_debug_infos.cend(),
                [](const DebugInfoPtr &info) { info->inlined_ = true; });
  auto debug_info_copy = copy_debug_infos.front();
  auto concat_pre_copy = copy_debug_infos.back();
  MS_LOG(DEBUG) << "debug_info_copy source lines:\n" << mindspore::ToString(trace::GetSourceLineList(debug_info_copy));
  MS_LOG(DEBUG) << "concat_pre_copy source lines:\n" << mindspore::ToString(trace::GetSourceLineList(concat_pre_copy));
  concat_pre_copy->set_trace_info(std::make_shared<TraceOpt>(concat));
  MS_LOG(DEBUG) << "New debug info source lines:\n" << mindspore::ToString(trace::GetSourceLineList(debug_info_copy));
  return debug_info_copy;
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

DebugInfoPtr NodeDebugInfo::Copy() {
  auto new_debug_info = std::make_shared<NodeDebugInfo>();
  new_debug_info->node_ = node_;
  new_debug_info->location_ = location_;
  new_debug_info->trace_info_ = trace_info_;
  new_debug_info->id_ = id_;
  new_debug_info->name_ = name_;
  new_debug_info->unique_id_ = unique_id_;
  new_debug_info->through_copy_unique_id_ = through_copy_unique_id_;
  new_debug_info->inlined_ = inlined_;
  return new_debug_info;
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
