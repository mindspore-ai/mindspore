/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include <utility>
#include <fstream>
#include <sstream>
#include <climits>
#include "ir/anf.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
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
std::string Location::ToString(SourceLineTip tip) {
  std::stringstream debug_info_ss;
  debug_info_ss << "In file " << file_name_ << "(" << line_ << ")" << std::endl;
  if (line_ <= 0) {
    return debug_info_ss.str();
  }

  char path[PATH_MAX + 1] = {0x00};
#if defined(_WIN32) || defined(_WIN64)
  if (file_name_.size() > PATH_MAX || _fullpath(path, file_name_.c_str(), PATH_MAX) == nullptr) {
    return debug_info_ss.str();
  }
#else
  if (file_name_.size() > PATH_MAX || realpath(file_name_.c_str(), path) == nullptr) {
    return debug_info_ss.str();
  }
#endif
  auto src_path = std::string(path);
  std::ifstream file(src_path);
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

void TraceContext::ProcessAttributeFromContext() {
  trace_info_ = nullptr;
  location_ = nullptr;
  func_name_ = "";
  // if there is trace context, get info from previous context
  if (!TraceManager::trace_context_stack_.empty()) {
    TraceContextPtr top = TraceManager::trace_context_stack_.top();
    trace_info_ = top->trace_info_;
    location_ = top->location_;
    func_name_ = top->func_name_;
  }
}

DebugInfo::DebugInfo() {
  InitValueFromContext();
  unique_id_ = gen_unique_id();
  debug_id_ = -1;
  name_ = "";
}

DebugInfo::DebugInfo(const std::string &name) {
  InitValueFromContext();
  unique_id_ = gen_unique_id();
  debug_id_ = -1;
  name_ = name;
}

DebugInfo::DebugInfo(const LocationPtr &loc) {
  InitValueFromContext();
  unique_id_ = gen_unique_id();
  debug_id_ = -1;
  location_ = loc;
}

int64_t DebugInfo::debug_id() {
  // cppcheck-suppress variableScope
  static int64_t cur_debug_id = 0;
  if (debug_id_ == -1) {
    debug_id_ = cur_debug_id;
    cur_debug_id++;
  }
  return debug_id_;
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

std::string DebugInfo::debug_name() {
  if (!name_.empty()) {
    return name_;
  }
  std::string debug_name = std::to_string(debug_id());
  name_ = debug_name;
  return debug_name;
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
  name_ = prefix + DebugInfo::debug_name();
  return name_;
}

std::string GraphDebugInfo::debug_name() {
  std::string prefix = "";
  return prefix + DebugInfo::debug_name();
}

LocationPtr GraphDebugInfo::location() {
  // function may have decorator which is included in its location
  if (deco_loc_ != nullptr) {
    LocationPtr loc = std::make_shared<Location>(*DebugInfo::location());
    loc->set_line(loc->line() + (deco_loc_->line_end() - deco_loc_->line() + 1));
    return loc;
  }
  return DebugInfo::location();
}
void GraphDebugInfo::set_deco_location(const LocationPtr &deco_list_loc) { deco_loc_ = deco_list_loc; }

TraceContextPtr TraceManager::CurrentContextInfo() {
  if (!TraceManager::trace_context_stack_.empty()) {
    return TraceManager::trace_context_stack_.top();
  }
  return nullptr;
}

void TraceManager::DebugTrace(const std::string &func_name, const LocationPtr &location) {
  TraceContextPtr context = std::make_shared<TraceContext>(location);
  context->set_func_name(func_name);
  TraceManager::trace_context_stack_.push(context);
}

void TraceManager::DebugTrace(const LocationPtr &location) {
  TraceContextPtr context = std::make_shared<TraceContext>(location);
  TraceManager::trace_context_stack_.push(context);
  TraceManager::parse_or_resolve_debug_info_ = std::make_shared<DebugInfo>(location);
}

void TraceManager::DebugTrace(const TraceInfoPtr &trace_info) {
  if (trace_info == nullptr) {
    MS_LOG(EXCEPTION) << "DebugTrace wrong traced info is null";
  }
  TraceContextPtr context = std::make_shared<TraceContext>(trace_info);
  if (trace_info->debug_info() == nullptr) {
    MS_LOG(EXCEPTION) << "Trace debug info is null";
  }
  TraceManager::trace_context_stack_.push(context);
  TraceManager::parse_or_resolve_debug_info_ = trace_info->debug_info();
}

void TraceManager::DebugTrace(const DebugInfoPtr &debug_info, const TraceInfoPtr &trace_info) {
  if (trace_info == nullptr) {
    MS_LOG(EXCEPTION) << "DebugTrace wrong traced info is null";
  }
  auto cloned_info = trace_info->clone();
  cloned_info->set_debug_info(debug_info);
  if (cloned_info->debug_info() == nullptr) {
    MS_LOG(EXCEPTION) << "Trace debug info is null with cloned trace";
  }
  TraceContextPtr context = std::make_shared<TraceContext>(cloned_info);
  TraceManager::trace_context_stack_.push(context);
}

void TraceManager::EndTrace() { TraceManager::trace_context_stack_.pop(); }

DebugInfoPtr TraceManager::GetParseOrResolveDebugInfo() { return TraceManager::parse_or_resolve_debug_info_; }

void TraceManager::ClearParseOrResolveDebugInfo() { TraceManager::parse_or_resolve_debug_info_ = nullptr; }

std::stack<TraceContextPtr> TraceManager::trace_context_stack_;

DebugInfoPtr TraceManager::parse_or_resolve_debug_info_ = nullptr;
}  // namespace mindspore
