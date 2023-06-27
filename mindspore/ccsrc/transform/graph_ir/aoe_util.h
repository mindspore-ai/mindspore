/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_AOE_UTIL_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_AOE_UTIL_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include "include/transform/graph_ir/types.h"

namespace mindspore {
namespace transform {
class AoeUtil {
 public:
  Status AoeOnlineGeGraph(std::shared_ptr<::ge::Session> ge_session, const transform::DfGraphPtr &graph);
  static AoeUtil &GetInstance();
  ~AoeUtil();
  void Initialize();
  void Destroy();
  void SaveOptimizedGraph(const int32_t &graph_id);
  bool IsSaveOptimizedGraph(const int32_t &graph_id) const;
  void RemoveWaitOptimizedGraph(const std::set<std::string> &optimized_graph_names);
  void AddOptimizeGraph(const std::string &graph_name);
  std::set<std::string> GetWaitOptimizeGraph() const;

 private:
  std::set<std::string> wait_optimize_graphs_;
  std::set<int32_t> optimized_graphs_id_;
  AoeUtil();
  bool initialize_;
  Status AoeGeGraph(::ge::Session *ge_session, const transform::DfGraphPtr &graph,
                    const std::map<::ge::AscendString, ::ge::AscendString> &tuningOptions);
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_AOE_UTIL_H_
