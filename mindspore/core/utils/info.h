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

#ifndef MINDSPORE_CORE_UTILS_INFO_H_
#define MINDSPORE_CORE_UTILS_INFO_H_

#include <iostream>
#include <string>
#include <memory>
#include <stack>
#include <utility>
#include <vector>

#include "base/base.h"
#include "utils/trace_info.h"

namespace mindspore {
// namespace to support intermediate representation definition
enum SourceLineTip { kSourceLineTipDiscard = 0, kSourceLineTipNextLine = 1, kSourceLineTipInLine = 2 };

// Location class record the location in source code.
class Location {
 public:
  Location(const std::string &file_name, int line, int column, int line_end, int column_end)
      : file_name_(file_name), line_(line), column_(column), line_end_(line_end), column_end_(column_end) {}
  Location(const Location &loc)
      : file_name_(loc.file_name_),
        line_(loc.line_),
        column_(loc.column_),
        line_end_(loc.line_end_),
        column_end_(loc.column_end_) {}
  std::string ToString(SourceLineTip tip = kSourceLineTipNextLine);
  std::string file_name() { return file_name_; }
  int line() const { return line_; }
  void set_line(int line) { line_ = line; }
  int line_end() const { return line_end_; }
  void set_line_end(int line) { line_end_ = line; }
  int column() const { return column_; }
  void set_column(int column) { column_ = column; }
  int column_end() const { return column_end_; }
  void set_column_end(int column) { column_end_ = column; }
  ~Location() = default;

 private:
  std::string file_name_;
  int line_;
  int column_;
  int line_end_;
  int column_end_;
};
class TraceContext;
using TraceContextPtr = std::shared_ptr<TraceContext>;

class TraceManager {
 public:
  TraceManager() = default;
  ~TraceManager() = default;
  static TraceContextPtr CurrentContextInfo();
  static void DebugTrace(const std::string &func_name, const LocationPtr &location);
  static void DebugTrace(const LocationPtr &location);
  static void DebugTrace(const TraceInfoPtr &trace_info);
  // debug trace with a cloned trace info with debug_info
  static void DebugTrace(const DebugInfoPtr &debug_info, const TraceInfoPtr &trace_info);
  static void EndTrace();

  static void ClearParseOrResolveDebugInfo();
  static DebugInfoPtr GetParseOrResolveDebugInfo();

  static std::stack<TraceContextPtr> trace_context_stack_;
  static DebugInfoPtr parse_or_resolve_debug_info_;
};

class TraceGuard {
 public:
  TraceGuard(const std::string func_name, const LocationPtr &location) {
    TraceManager::DebugTrace(func_name, location);
  }
  explicit TraceGuard(const LocationPtr &location) { TraceManager::DebugTrace(location); }
  explicit TraceGuard(const TraceInfoPtr &trace_info) { TraceManager::DebugTrace(trace_info); }
  TraceGuard(const DebugInfoPtr &debug_info, const TraceInfoPtr &trace_info) {
    TraceManager::DebugTrace(debug_info, trace_info);
  }
  ~TraceGuard() { TraceManager::EndTrace(); }
};

class TraceContext {
 public:
  LocationPtr location_;
  TraceInfoPtr trace_info_;
  std::string func_name_;

 protected:
  void ProcessAttributeFromContext();

 public:
  ~TraceContext() = default;
  explicit TraceContext(const LocationPtr &loc) {
    ProcessAttributeFromContext();
    location_ = loc;
  }
  explicit TraceContext(const std::string &func_name) {
    ProcessAttributeFromContext();
    func_name_ = func_name;
  }
  explicit TraceContext(const TraceInfoPtr &trace_info) {
    ProcessAttributeFromContext();
    trace_info_ = trace_info;
  }
  void set_location(const LocationPtr &loc) { location_ = loc; }
  LocationPtr location() { return location_; }
  void set_trace_info(const TraceInfoPtr &trace_info) { trace_info_ = trace_info; }
  TraceInfoPtr trace_info() const { return trace_info_; }
  void set_func_name(const std::string &func_name) { func_name_ = func_name; }
  std::string func_name() { return func_name_; }
};

class DebugInfo : public Base {
 public:
  DebugInfo();

  explicit DebugInfo(const std::string &name);

  explicit DebugInfo(const LocationPtr &loc);

  ~DebugInfo() override = default;
  MS_DECLARE_PARENT(DebugInfo, Base);
  int64_t debug_id();
  int64_t unique_id() const { return unique_id_; }
  int64_t unique_id_through_copy() const;
  std::string get_id() { return std::to_string(debug_id()); }

  void set_trace_info(const TraceInfoPtr &trace_info) { trace_info_ = trace_info; }
  TraceInfoPtr trace_info() const { return trace_info_; }
  void set_location(const LocationPtr &loc) { location_ = loc; }
  virtual LocationPtr location() { return location_; }
  std::string name() { return name_; }
  void set_name(const std::string &name) { name_ = name; }
  virtual std::string debug_name();

  virtual std::string get_python_func_belonged() { return ""; }

 protected:
  template <typename Derived>
  std::shared_ptr<Derived> shared_from_base() {
    return std::static_pointer_cast<Derived>(shared_from_this());
  }

 private:
  void InitValueFromContext() {
    if (TraceManager::CurrentContextInfo() != nullptr) {
      auto context_info = TraceManager::CurrentContextInfo();
      trace_info_ = context_info->trace_info();
      location_ = context_info->location();
    }
  }
  static int64_t gen_unique_id() {
    static int64_t cur_unique_id = 0;
    return cur_unique_id++;
  }

 protected:
  int64_t unique_id_;
  int64_t debug_id_;
  TraceInfoPtr trace_info_;
  LocationPtr location_;
  std::string name_;
};

class NodeDebugInfo : public DebugInfo {
 public:
  NodeDebugInfo() {
    if (TraceManager::CurrentContextInfo() != nullptr) {
      auto context_info = TraceManager::CurrentContextInfo();
      py_func_belonged_ = context_info->func_name();
    }
  }
  explicit NodeDebugInfo(const std::string &name) : DebugInfo(name) {
    if (TraceManager::CurrentContextInfo() != nullptr) {
      auto context_info = TraceManager::CurrentContextInfo();
      py_func_belonged_ = context_info->func_name();
    }
  }
  ~NodeDebugInfo() override = default;

  std::string debug_name() override;
  void set_node(const std::shared_ptr<AnfNode> &node) { node_ = AnfNodeWeakPtr(node); }
  std::shared_ptr<AnfNode> get_node() const { return node_.lock(); }
  void set_py_func_belonged(const std::string &name) { py_func_belonged_ = name; }
  std::string get_python_func_belonged() override { return py_func_belonged_; }
  AnfNodeWeakPtr node_;
  std::string py_func_belonged_;
};
using NodeDebugInfoPtr = std::shared_ptr<NodeDebugInfo>;

class GraphDebugInfo : public DebugInfo {
 public:
  GraphDebugInfo() {
    if (TraceManager::CurrentContextInfo() != nullptr) {
      auto context_info = TraceManager::CurrentContextInfo();
      py_func_name_ = context_info->func_name();
      deco_loc_ = nullptr;
    }
  }

  explicit GraphDebugInfo(const std::string &name) : DebugInfo(name) {
    if (TraceManager::CurrentContextInfo() != nullptr) {
      auto context_info = TraceManager::CurrentContextInfo();
      py_func_name_ = context_info->func_name();
      deco_loc_ = nullptr;
    }
  }
  ~GraphDebugInfo() override = default;
  std::string debug_name() override;
  LocationPtr location() override;
  LocationPtr deco_location() { return deco_loc_; }
  void set_graph(const FuncGraphPtr &func_graph) { func_graph_ = FuncGraphWeakPtr(func_graph); }
  FuncGraphPtr get_graph() const { return func_graph_.lock(); }
  void set_full_name(const std::string &name) { full_name_ = name; }
  std::string get_full_name() { return full_name_; }
  void set_deco_location(const LocationPtr &deco_list_loc);
  std::string get_python_func_belonged() override { return py_func_name_; }
  FuncGraphWeakPtr func_graph_;
  LocationPtr deco_loc_;
  std::string py_func_name_;
  std::string full_name_;
};

using GraphDebugInfoPtr = std::shared_ptr<GraphDebugInfo>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_INFO_H_
