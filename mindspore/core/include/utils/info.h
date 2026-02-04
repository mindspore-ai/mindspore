/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include <string>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "base/base.h"
#include "mindapi/base/macros.h"
#include "utils/log_adapter.h"

namespace mindspore {
enum SourceLineTip {
  kSourceLineTipDiscard = 0,
  kSourceLineTipNextLine = 1,
  kSourceLineTipInLine = 2,
  kSourceSectionTipNextLineHere = 3
};

// typedef enum CacheBool { UNCACHED = -1, FALSE, TRUE } CacheBool;
using CacheBool = int32_t;
const CacheBool Uncached = -1;
const CacheBool False = 0;
const CacheBool True = 1;

MS_CORE_API void ClearThreadLocal();

class TraceInfo;
using TraceInfoPtr = std::shared_ptr<TraceInfo>;
class Location;
using LocationPtr = std::shared_ptr<Location>;
class DebugInfo;
using DebugInfoPtr = std::shared_ptr<DebugInfo>;

// Location class record the location in source code.
class Location {
 public:
  Location(const std::string &file_name, int line, int column, int line_end, int column_end, const std::string &expr,
           std::vector<std::string> &&comments)
      : file_name_(file_name),
        line_(line),
        column_(column),
        line_end_(line_end),
        column_end_(column_end),
        expr_src_(expr),
        comments_(std::move(comments)) {}
  ~Location() = default;
  MS_CORE_API std::string ToString(SourceLineTip tip = kSourceLineTipNextLine, int start_line = 0);
  MS_CORE_API std::string GetStartLineSourceCode(SourceLineTip tip = kSourceLineTipNextLine);
  MS_CORE_API std::string DebugString() const;
  std::string file_name() const { return file_name_; }
  void set_file_name(const std::string &file_name) { file_name_ = file_name; }
  int line() const { return line_; }
  void set_line(int line) { line_ = line; }
  int line_end() const { return line_end_; }
  void set_line_end(int line_end) { line_end_ = line_end; }
  int column() const { return column_; }
  int column_end() const { return column_end_; }
  const std::string &expr_src() const { return expr_src_; }
  void set_expr_src(const std::string &expr_src) { expr_src_ = expr_src; }
  const std::vector<std::string> &comments() const { return comments_; }

  bool invalid() const { return line() == 0 && line_end() == 0 && column() == 0 && column_end() == 0; }

  bool operator<(const Location &other) const;

 private:
  bool ReadSectionDebugInfoFromFile(SourceLineTip tip, int start_line, std::stringstream &section_debug_info_ss);

  std::string file_name_;
  int line_;
  int column_;
  int line_end_;
  int column_end_;
  std::string expr_src_;
  std::vector<std::string> comments_;
  std::string line_str_;
};

class MS_CORE_API TraceContext {
 public:
  explicit TraceContext(const LocationPtr &loc);
  explicit TraceContext(const TraceInfoPtr &trace_info);
  ~TraceContext() = default;
  const LocationPtr &location() const { return location_; }
  const TraceInfoPtr &trace_info() const { return trace_info_; }

 private:
  LocationPtr location_;
  TraceInfoPtr trace_info_;
};

using TraceContextPtr = TraceContext *;

/// \brief TraceManager defines interface for debug trace management.
class MS_CORE_API TraceManager {
 public:
  /// \brief Constructor of TraceManager.
  TraceManager() = default;

  /// \brief Destructor of TraceManager.
  ~TraceManager() = default;

  /// \brief Get current trace context.
  ///
  /// \return The current trace context.
  static TraceContextPtr CurrentContextInfo();

  /// \brief Get the call stack of all trace contexts.
  ///
  /// \return All trace contexts in the call stack, with the top-of-stack element at the end of the std::vector.
  static const std::vector<TraceContext> &trace_context_stack();

  /// \brief Debug trace with the given location.
  ///
  /// \param[in] location The source code location for debug trace.
  /// \return If trace successfully.
  static bool DebugTrace(const LocationPtr &location);

  /// \brief Debug trace with the given trace info.
  ///
  /// \param[in] trace_info The trace info for debug.
  /// \return If trace successfully.
  static bool DebugTrace(const TraceInfoPtr &trace_info);

  /// \brief End current debug trace.
  static void EndTrace() noexcept;

  /// \brief Clear debug info for parse or resolve.
  static void ClearParserDebugInfo();

  /// \brief Get debug info for parse or resolve.
  ///
  /// \return The debug info for parse or resolve.
  static DebugInfoPtr parser_debug_info();

  /// \brief Get the flag of recording a debug info.
  ///
  /// \return A bool.
  static bool parser_debug_info_flag();

  /// \brief Set the flag to false for not recording a debug info.
  static void CloseParserDebugInfoFlag();

  /// \brief Set the flag to true for recording a debug info.
  static void OpenParserDebugInfoFlag();
};

class TraceGuard {
 public:
  explicit TraceGuard(const LocationPtr &location) { tracing_ = TraceManager::DebugTrace(location); }
  explicit TraceGuard(const TraceInfoPtr &trace_info) { tracing_ = TraceManager::DebugTrace(trace_info); }
  ~TraceGuard() {
    if (tracing_) {
      TraceManager::EndTrace();
    }
  }

 private:
  bool tracing_{false};
};

/// \brief DebugInfo defines information for debug trace.
class MS_CORE_API DebugInfo {
 public:
  /// \brief Construct a default DebugInfo.
  DebugInfo() : DebugInfo("") {}

  /// \brief Construct DebugInfo with the given name.
  ///
  /// \param[in] name The DebugInfo name.
  explicit DebugInfo(const std::string &name);

  /// \brief Construct DebugInfo with the given location.
  ///
  /// \param[in] loc The location for DebugInfo.
  explicit DebugInfo(const LocationPtr &loc);

  /// \brief Construct DebugInfo with the given trace info.
  ///
  /// \param[in] trace_info The trace info for DebugInfo.
  explicit DebugInfo(TraceInfoPtr &&trace_info) : unique_id_(gen_unique_id()), trace_info_(std::move(trace_info)) {}

  /// \brief Destructor of DebugInfo.
  virtual ~DebugInfo() = default;

  /// \brief Get the id.
  ///
  /// \return The id of the debug info.
  size_t get_id() const;

  /// \brief Get the unique id.
  ///
  /// \return The unique id.
  size_t unique_id() const { return unique_id_; }

  /// \brief Get the unique id through copy.
  ///
  /// \return The unique id through copy.
  size_t unique_id_through_copy() const;

  /// \brief Set the trace info.
  ///
  /// \param[in] trace_info The trace info to be set.
  void set_trace_info(const TraceInfoPtr &trace_info) { trace_info_ = trace_info; }

  /// \brief Get the trace info.
  ///
  /// \return The trace info.
  TraceInfoPtr trace_info() const { return trace_info_; }

  /// \brief Set the location.
  ///
  /// \param[in] loc The location to be set.
  void set_location(const LocationPtr &loc) { location_ = loc; }

  /// \brief Get the location.
  ///
  /// \return The location.
  virtual LocationPtr location() const { return location_; }

  /// \brief Get the name.
  ///
  /// \return The name of the DebugInfo.
  std::string name() const { return name_; }

  /// \brief Set the name.
  ///
  /// \param[in] name The name to be set.
  void set_name(const std::string &name) { name_ = name; }

  bool is_reusing() const { return is_reusing_; }

  void set_is_reusing(bool reuse = true) { is_reusing_ = reuse; }

  /// \brief Get the debug name.
  ///
  /// \return The debug name of the DebugInfo.
  virtual std::string debug_name() { return debug_name_; }

  /// \brief Set the debug name.
  ///
  /// \param[in] debug_name The name to be set.
  void set_debug_name(const std::string &debug_name) { debug_name_ = debug_name; }

  HashMap<DebugInfoPtr, DebugInfoPtr> &shadow_debug_infos_map() { return shadow_debug_infos_map_; }

  const std::vector<DebugInfoPtr> &real_loc() { return real_loc_; }

  void AddLocation(const DebugInfoPtr &debug_info) { (void)real_loc_.emplace_back(debug_info); }

  void set_real_loc(const std::vector<DebugInfoPtr> &debug_infos) {
    (void)real_loc_.insert(real_loc_.cend(), debug_infos.cbegin(), debug_infos.cend());
  }

  static DebugInfoPtr UpdateInlineCNodeDebugInfo(const DebugInfoPtr &call_debug_info, const DebugInfoPtr &debug_info);

 protected:
  static size_t gen_unique_id() {
    static size_t cur_unique_id = 0;
    return cur_unique_id++;
  }

  mutable size_t id_ = 0;
  size_t unique_id_;
  size_t through_copy_unique_id_{std::numeric_limits<size_t>::max()};
  TraceInfoPtr trace_info_;
  LocationPtr location_;
  std::string name_;
  std::string debug_name_;
  HashMap<DebugInfoPtr, DebugInfoPtr> shadow_debug_infos_map_;
  std::vector<DebugInfoPtr> real_loc_;
  bool is_reusing_ = false;
};

/// \brief NodeDebugInfo defines debug information for a node.
class MS_CORE_API NodeDebugInfo : public DebugInfo {
 public:
  /// \brief Construct a default NodeDebugInfo.
  NodeDebugInfo() : DebugInfo() {}

  /// \brief Construct NodeDebugInfo with a given name.
  ///
  /// \param[in] name the name of the NodeDebugInfo.
  explicit NodeDebugInfo(const std::string &name) : DebugInfo(name) {}

  /// \brief Construct NodeDebugInfo with the given trace info.
  ///
  /// \param[in] trace_info The trace info for NodeDebugInfo.
  explicit NodeDebugInfo(TraceInfoPtr &&trace_info) : DebugInfo(std::move(trace_info)) {}

  /// \brief Destructor of the NodeDebugInfo.
  ~NodeDebugInfo() override = default;

  std::string debug_name() override;

  /// \brief Set the node's type name.
  ///
  /// \param[in] type_name The node type name to be set.
  void set_type_name(const std::string &type_name) { type_name_ = type_name; }

 private:
  std::string type_name_;
};

using NodeDebugInfoPtr = std::shared_ptr<NodeDebugInfo>;

class MS_CORE_API GraphDebugInfo : public DebugInfo {
 public:
  GraphDebugInfo() : DebugInfo() {}

  explicit GraphDebugInfo(const std::string &name) : DebugInfo(name) {}

  explicit GraphDebugInfo(TraceInfoPtr &&trace_info) : DebugInfo(std::move(trace_info)) {}

  ~GraphDebugInfo() override = default;

  std::string debug_name() override;
  LocationPtr location() const override;
  LocationPtr deco_location() { return deco_loc_; }
  void set_graph(const FuncGraphPtr &func_graph) { func_graph_ = FuncGraphWeakPtr(func_graph); }
  FuncGraphPtr get_graph() const { return func_graph_.lock(); }
  void set_deco_location(const LocationPtr &deco_list_loc);

 private:
  FuncGraphWeakPtr func_graph_;
  LocationPtr deco_loc_;
};

using GraphDebugInfoPtr = std::shared_ptr<GraphDebugInfo>;

struct MS_CORE_API DebugInfoCompare {
  bool operator()(const DebugInfoPtr &left, const DebugInfoPtr &right) const;
};

void UpdateInlineCNodeDebugInfo(const AnfNodePtr &caller, const AnfNodePtr &callee);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_INFO_H_
