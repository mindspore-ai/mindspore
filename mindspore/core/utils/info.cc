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
#include "utils/compile_config.h"

namespace mindspore {
namespace {
/// \brief Trace context stack for current thread.
thread_local std::vector<TraceContext> trace_context_stack_;

/// \brief Record a debug info for print.
thread_local DebugInfoPtr parser_debug_info_ = nullptr;

/// \brief A flag to decide whether record a debug info or not.
thread_local bool parser_debug_info_flag_ = false;
}  // namespace

void ClearThreadLocal() {
  trace_context_stack_.clear();
  parser_debug_info_.reset();
}

std::string HighlightLine(const std::string &line, int col_begin, int col_end, bool single_line, SourceLineTip tip) {
  if (col_begin < col_end && col_begin != -1 && col_end <= SizeToLong(line.length()) && tip != kSourceLineTipDiscard) {
    std::stringstream oss;
    if (tip == kSourceLineTipInLine) {
      if (single_line) {
        std::string start = line.substr(0, LongToSize(col_begin));
        std::string trimmed = line.substr(LongToSize(col_begin), LongToSize(col_end - col_begin));
        std::string end = line.substr(LongToSize(col_end), LongToSize(SizeToLong(line.length()) - col_end));
        oss << start << "<" << trimmed << ">" << end << "\n";
      } else {
        oss << line << "\n";
      }
    } else if (tip == kSourceLineTipNextLine) {
      std::string start = line.substr(0, LongToSize(col_begin));
      std::string start_spaces(start.length(), ' ');
      oss << line << "\n" << start_spaces << "^";
      if (single_line) {
        for (size_t i = 1; i < LongToSize(col_end - col_begin); ++i) {
          oss << "~";
        }
      }
    } else if (tip == kSourceSectionTipNextLineHere) {
      std::string start = line.substr(0, LongToSize(col_begin));
      std::string start_spaces(start.length(), ' ');
      oss << line << "\n" << start_spaces << "~<-------------HERE";
    }
    return oss.str();
  }
  return line;
}

std::string Location::DebugString() const {
  std::stringstream ss;
  ss << "Location{\n";
  ss << "\tfile_name_: " << file_name_ << ",\n";
  ss << "\tline_: " << line_ << ",\n";
  ss << "\tline_end_: " << line_end_ << ",\n";
  ss << "\tcolumn_: " << column_ << ",\n";
  ss << "\tcolumn_end_: " << column_end_ << ",\n";
  ss << "\tcomments_: " << comments_ << "\n";
  ss << "}\n";
  return ss.str();
}

// Generate debug information for the location node .
// print the file name, line no and column no, and part of the content
std::string Location::ToString(SourceLineTip tip, int start_line) {
  std::stringstream debug_info_ss;
  std::stringstream section_debug_info_ss;
  if (tip != kSourceSectionTipNextLineHere) {
    // For example,
    // the location is from {line 9, column 4}, to {line 15, column 20}:
    //     In file /x/xxx/x.py:9~15, 4~20
    // If in single line, from {line 9, column 4}, to {line 9, column 20}:
    //     In file /x/xxx/x.py:9, 4~20
    debug_info_ss << "In file " << file_name_ << ":" << line_;
    if (line_ <= 0) {
      return debug_info_ss.str();
    }
    if (line_end_ > line_) {
      debug_info_ss << "~" << line_end_;
    }
    debug_info_ss << ", " << column_ << "~" << column_end_ << std::endl;
    // Use line_str_ as cache.
    if (!line_str_.empty()) {
      debug_info_ss << HighlightLine(line_str_, column_, column_end_, line_end_ == line_, tip) << std::endl;
      return debug_info_ss.str();
    }
  } else {  // tip == kSourceSectionTipNextLineHere
    section_debug_info_ss << "In file " << file_name_ << ":" << line_ << std::endl;
  }

  // Start read the specific line. Optimize here by seekg().
  auto path = FileUtils::GetRealPath(file_name_.c_str());
  if (!path.has_value()) {
    MS_LOG(WARNING) << "The file '" << file_name_ << "' may not exists.";
    return debug_info_ss.str();
  }
  std::ifstream file(path.value());
  if (!file.is_open()) {
    MS_LOG(WARNING) << "Failed to open file '" << file_name_ << "'.";
    return debug_info_ss.str();
  }
  // Read the lines one by one.
  int line_num = 0;
  std::string line;
  (void)getline(file, line);
  while (line_num != line_ - 1) {
    if (tip == kSourceSectionTipNextLineHere && line_num >= start_line - 1) {
      section_debug_info_ss << line << "\n";
    }
    (void)getline(file, line);
    line_num++;
  }
  file.close();
  // Store the line string as cache.
  line_str_ = line;

  if (tip == kSourceSectionTipNextLineHere) {
    section_debug_info_ss << HighlightLine(line, column_, column_end_, line_end_ == line_, tip) << std::endl;
    return section_debug_info_ss.str();
  }
  debug_info_ss << HighlightLine(line, column_, column_end_, line_end_ == line_, tip) << std::endl;
  return debug_info_ss.str();
}

bool Location::operator<(const Location &other) const {
  auto ret = file_name_.compare(other.file_name());
  if (ret != 0) {
    return ret < 0;
  }
  return line_ < other.line();
}

DebugInfo::DebugInfo(const std::string &name) : unique_id_(gen_unique_id()), name_(name) {
  auto top = TraceManager::CurrentContextInfo();
  if (top != nullptr) {
    trace_info_ = top->trace_info();
    location_ = top->location();
  }
}

DebugInfo::DebugInfo(const LocationPtr &loc) : unique_id_(gen_unique_id()), location_(loc) {
  auto top = TraceManager::CurrentContextInfo();
  if (top != nullptr) {
    trace_info_ = top->trace_info();
  }
}

size_t DebugInfo::get_id() const {
  // cppcheck-suppress variableScope
  static size_t current_id = 0;
  if (id_ == 0) {
    id_ = ++current_id;
  }
  return id_;
}

size_t DebugInfo::unique_id_through_copy() const {
  auto info = trace_info();
  auto final_unique_id =
    through_copy_unique_id_ == std::numeric_limits<size_t>::max() ? unique_id_ : through_copy_unique_id_;
  if (info != nullptr) {
    if (info->isa<TraceCopy>() && info->debug_info() != nullptr) {
      final_unique_id = info->debug_info()->unique_id_through_copy();
    }
  }
  const_cast<DebugInfo *>(this)->through_copy_unique_id_ = final_unique_id;
  return final_unique_id;
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

std::string NodeDebugInfo::debug_name() {
  if (!debug_name_.empty()) {
    return debug_name_;
  }
  std::ostringstream oss;
  oss << type_name_ << "_" << get_id();
  debug_name_ = oss.str();
  return debug_name_;
}

std::string GraphDebugInfo::debug_name() {
  if (debug_name_.empty()) {
    debug_name_ = "_anonymous_";  // Represent <anonymous>
  }
  return debug_name_;
}

LocationPtr GraphDebugInfo::location() const {
  // Function may have decorator which is included in its location.
  auto loc = DebugInfo::location();
  if (deco_loc_ != nullptr && loc != nullptr) {
    auto loc_line = loc->line() + ((deco_loc_->line_end() - deco_loc_->line()) + 1);
    auto comments = loc->comments();
    return std::make_shared<Location>(loc->file_name(), loc_line, loc->line_end(), loc->column(), loc->column_end(),
                                      loc->expr_src(), std::move(comments));
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

bool TraceManager::DebugTrace(const LocationPtr &location) {
  MS_EXCEPTION_IF_NULL(location);
  if (location->invalid()) {
    MS_LOG(DEBUG) << "Trace failed";
    return false;
  }
  (void)trace_context_stack_.emplace_back(location);
  if (parser_debug_info_flag_) {
    parser_debug_info_ = std::make_shared<DebugInfo>(location);
  }
  MS_LOG(DEBUG) << "location: " << location->ToString();
  return true;
}

bool TraceManager::DebugTrace(const TraceInfoPtr &trace_info) {
  MS_EXCEPTION_IF_NULL(trace_info);
  auto &debug_info = trace_info->debug_info();
  MS_EXCEPTION_IF_NULL(debug_info);
  (void)trace_context_stack_.emplace_back(trace_info);
  if (parser_debug_info_flag_) {
    parser_debug_info_ = debug_info;
  }
  return true;
}

void TraceManager::EndTrace() noexcept { trace_context_stack_.pop_back(); }

DebugInfoPtr TraceManager::parser_debug_info() { return parser_debug_info_; }

void TraceManager::ClearParserDebugInfo() { parser_debug_info_ = nullptr; }

void TraceManager::CloseParserDebugInfoFlag() {
  parser_debug_info_flag_ = false;
  ClearParserDebugInfo();
}

void TraceManager::OpenParserDebugInfoFlag() { parser_debug_info_flag_ = true; }

bool TraceManager::parser_debug_info_flag() { return parser_debug_info_flag_; }

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

namespace {
void DumpNodesDebugInfos(const AnfNodePtr &caller, const AnfNodePtr &callee) {
  const DebugInfoPtr &caller_debug_info = caller->debug_info();
  const DebugInfoPtr &callee_debug_info = callee->debug_info();
  const auto caller_debug_infos = GetDebugInfoList(caller_debug_info);
  const auto callee_debug_infos = GetDebugInfoList(callee_debug_info);
  MS_LOG(ERROR) << "caller: " << caller << "/" << caller->DebugString() << ", caller_debug_info: " << callee << "/"
                << caller_debug_info << ", debug info size: " << caller_debug_infos.size();
  MS_LOG(ERROR) << "callee: " << callee->DebugString() << ", callee_debug_info: " << callee_debug_info
                << ", debug info size: " << callee_debug_infos.size();
  for (size_t i = 0; i < caller_debug_infos.size(); ++i) {
    MS_LOG(ERROR) << "# caller_debug_infos[" << i << "]: " << caller_debug_infos[i] << "/"
                  << caller_debug_infos[i]->name() << "/" << caller_debug_infos[i]->debug_name() << "/"
                  << trace::GetDebugInfoStr(caller_debug_infos[i], "", kSourceLineTipNextLine, true) << ", trace: "
                  << (caller_debug_infos[i]->trace_info() != nullptr ? caller_debug_infos[i]->trace_info()->name()
                                                                     : "none");
  }
  for (size_t i = 0; i < callee_debug_infos.size(); ++i) {
    MS_LOG(ERROR) << "# callee_debug_infos[" << i << "]: " << callee_debug_infos[i] << "/"
                  << callee_debug_infos[i]->name() << "/" << callee_debug_infos[i]->debug_name() << "/"
                  << trace::GetDebugInfoStr(callee_debug_infos[i], "", kSourceLineTipNextLine, true) << ", trace: "
                  << (callee_debug_infos[i]->trace_info() != nullptr ? callee_debug_infos[i]->trace_info()->name()
                                                                     : "none");
  }
}

void SyncShadowDebugInfo(const DebugInfoPtr &caller_debug_info, const DebugInfoPtr &callee_debug_info) {
  // Synchronize callers' shadow debug infos.
  const auto &caller_shadow_debug_infos = caller_debug_info->shadow_debug_infos_map();
  callee_debug_info->shadow_debug_infos_map().insert(caller_shadow_debug_infos.cbegin(),
                                                     caller_shadow_debug_infos.cend());
}
}  // namespace

void UpdateInlineCNodeDebugInfo(const AnfNodePtr &caller, const AnfNodePtr &callee) {
  const DebugInfoPtr &caller_debug_info = caller->debug_info();
  const DebugInfoPtr &callee_debug_info = callee->debug_info();
  const auto caller_debug_infos = GetDebugInfoList(caller_debug_info);
  const auto callee_debug_infos = GetDebugInfoList(callee_debug_info);
  if (callee_debug_infos.size() == 1) {  // New inserted node, not by parser.
    SyncShadowDebugInfo(caller_debug_info, callee_debug_info);
    return;
  }
  int64_t pos = -1;
  for (size_t i = 0; i < caller_debug_infos.size() && i < callee_debug_infos.size(); ++i) {
    const auto &cur_caller_debug_info = caller_debug_infos[caller_debug_infos.size() - i - 1];
    const auto &cur_callee_debug_info = callee_debug_infos[callee_debug_infos.size() - i - 1];
    if (cur_caller_debug_info == cur_callee_debug_info) {
      continue;
    }
    const auto &caller_locaton = cur_caller_debug_info->location();
    const auto &callee_locaton = cur_callee_debug_info->location();
    if (caller_locaton == nullptr || callee_locaton == nullptr) {
      SyncShadowDebugInfo(caller_debug_info, callee_debug_info);
      return;
    }
    if (caller_locaton != callee_locaton) {
      pos = SizeToLong(i);
      break;
    }
  }
  if (pos == -1) {
    SyncShadowDebugInfo(caller_debug_info, callee_debug_info);
    return;
  }
  MS_LOG(DEBUG) << "pos: " << pos << ", caller_debug_info: " << caller_debug_info << "/"
                << trace::GetDebugInfoStr(caller_debug_info, "", kSourceLineTipNextLine, true)
                << ", callee_debug_info: " << callee_debug_info << "/"
                << trace::GetDebugInfoStr(callee_debug_info, "", kSourceLineTipNextLine, true);
  // Change the parse func debug info with call func debug info.
  const int64_t callee_reverse_pos = SizeToLong(callee_debug_infos.size()) - pos - 1;
  if (callee_reverse_pos < 0) {
    DumpNodesDebugInfos(caller, callee);
    MS_LOG(INTERNAL_EXCEPTION) << "Wrong index for callee.";
  }
  auto parse_def_debug_info = callee_debug_infos[callee_reverse_pos];
  MS_EXCEPTION_IF_NULL(parse_def_debug_info);
  const int64_t caller_reverse_pos = SizeToLong(caller_debug_infos.size()) - pos - 1;
  if (caller_reverse_pos < 0) {
    DumpNodesDebugInfos(caller, callee);
    MS_LOG(INTERNAL_EXCEPTION) << "Wrong index for caller.";
  }
  auto first_diff_caller_debug_info = caller_debug_infos[caller_reverse_pos];
  MS_LOG(DEBUG) << "reverse_pos: " << callee_reverse_pos << "/" << caller_reverse_pos
                << ", callee_debug_info: " << callee_debug_info << ", parse_def_debug_info: " << parse_def_debug_info
                << "/" << trace::GetDebugInfoStr(parse_def_debug_info, "", kSourceLineTipNextLine, true)
                << ", first_diff_caller_debug_info: " << first_diff_caller_debug_info << "/"
                << trace::GetDebugInfoStr(first_diff_caller_debug_info, "", kSourceLineTipNextLine, true);
  // Insert inlined shadow debug info pair.
  (void)callee_debug_info->shadow_debug_infos_map().emplace(parse_def_debug_info, first_diff_caller_debug_info);
  // Synchronize callers' shadow debug infos.
  SyncShadowDebugInfo(caller_debug_info, callee_debug_info);
}

std::vector<DebugInfoPtr> GetDebugInfoList(const DebugInfoPtr &debug_info) {
  std::vector<DebugInfoPtr> debug_infos;
  auto cur_debug_info = debug_info;
  while (cur_debug_info != nullptr) {
    (void)debug_infos.emplace_back(cur_debug_info);
    if (cur_debug_info->trace_info() == nullptr) {
      break;
    }
    cur_debug_info = cur_debug_info->trace_info()->debug_info();
  }
  return debug_infos;
}
}  // namespace mindspore
