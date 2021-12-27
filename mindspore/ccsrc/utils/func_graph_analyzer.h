/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_FUNC_GRAPH_ANALYZER_H_
#define MINDSPORE_CCSRC_UTILS_FUNC_GRAPH_ANALYZER_H_

#include <string>
#include <memory>
#include <vector>
#include "ir/func_graph.h"
namespace mindspore {
class ValueManager;
class FuncClosure {
 public:
  FuncClosure(const FuncGraphPtr &func_graph, const std::vector<size_t> &arg_indexes,
              const std::vector<CNodePtr> &arg_users)
      : func_graph_(func_graph), arg_indexes_(arg_indexes), arg_users_(arg_users) {}
  ~FuncClosure() = default;

  bool operator==(const FuncClosure &other) const {
    return func_graph_ == other.func_graph_ && arg_users_ == other.arg_users_ && arg_indexes_ == other.arg_indexes_;
  }

  bool ExistInList(const std::vector<std::shared_ptr<FuncClosure>> &list) const;

  std::vector<AnfNodePtr> GetArgs() const;

  std::string ToString() const;

  FuncGraphPtr func_graph_;
  std::vector<size_t> arg_indexes_;
  std::vector<CNodePtr> arg_users_;
};
using FuncClosurePtr = std::shared_ptr<FuncClosure>;

class FuncGraphAnalyzer {
 public:
  explicit FuncGraphAnalyzer(const FuncGraphPtr &func_graph);
  ~FuncGraphAnalyzer() = default;
  void Run();

  std::vector<CNodePtr> GetFuncGraphCallers(const FuncGraphPtr &func_graph) const;

  std::vector<FuncGraphPtr> GetCallerFuncGraphs(const AnfNodePtr &node) const;
  const std::vector<FuncClosurePtr> &GetCallClosures(const AnfNodePtr &call) const;
  // A call node call a same graph by different partial, parameter may have more than one closure so have more than one
  // args;
  std::vector<AnfNodePtr> GetArg(const AnfNodePtr &param, const AnfNodePtr &call) const;

  void DumpFuncGraphRealUsers() const;
  // If has partial and args are binded, return true, otherwise return false.
  bool ExistClosure() const;
  // If has undirect call, return true, other wise return false.
  bool HasIncorporateCall() const;

 private:
  FuncGraphPtr root_graph_;
  std::shared_ptr<ValueManager> value_manager_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_FUNC_GRAPH_ANALYZER_H_
