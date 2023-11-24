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
#include "aoe/external/aoe.h"
#include "include/transform/graph_ir/types.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace transform {
ORIGIN_METHOD(AoeInitialize, Aoe::AoeStatus, const std::map<::ge::AscendString, ::ge::AscendString> &);
ORIGIN_METHOD(AoeFinalize, Aoe::AoeStatus);
ORIGIN_METHOD(AoeCreateSession, Aoe::AoeStatus, uint64_t &);
ORIGIN_METHOD(AoeSetGeSession, Aoe::AoeStatus, uint64_t, ::ge::Session *);
ORIGIN_METHOD(AoeSetTuningGraph, Aoe::AoeStatus, uint64_t, const ::ge::Graph &);
ORIGIN_METHOD(AoeTuningGraph, Aoe::AoeStatus, uint64_t, const std::map<::ge::AscendString, ::ge::AscendString> &);
ORIGIN_METHOD(AoeDestroySession, Aoe::AoeStatus, uint64_t);

class AoeUtil {
 public:
  Status AoeOnlineGeGraph(const std::shared_ptr<::ge::Session> &ge_session, const transform::DfGraphPtr &graph) const;
  static AoeUtil &GetInstance();
  ~AoeUtil();
  void Initialize();
  void Destroy();
  void SaveOptimizedGraph(const int32_t &graph_id);
  bool IsSaveOptimizedGraph(const int32_t &graph_id) const;
  void RemoveWaitOptimizedGraph(const std::set<std::string> &optimized_graph_names);
  void AddOptimizeGraph(const std::string &graph_name);
  std::set<std::string> GetWaitOptimizeGraph() const;
  void SetOfflineEnvDumpGeGraph();

 private:
  std::set<std::string> wait_optimize_graphs_;
  std::set<int32_t> optimized_graphs_id_;
  AoeUtil();
  bool initialize_;

  Status AoeGeGraph(::ge::Session *ge_session, const transform::DfGraphPtr &graph,
                    const std::map<::ge::AscendString, ::ge::AscendString> &tuningOptions) const;
  void *plugin_handle_ = nullptr;
  AoeInitializeFunObj aoe_initialize_ = nullptr;
  AoeFinalizeFunObj aoe_finalize_ = nullptr;
  AoeCreateSessionFunObj aoe_create_session_ = nullptr;
  AoeSetGeSessionFunObj aoe_set_ge_gession_ = nullptr;
  AoeSetTuningGraphFunObj aoe_set_tuning_graph_ = nullptr;
  AoeTuningGraphFunObj aoe_tuning_graph_ = nullptr;
  AoeDestroySessionFunObj aoe_destroy_session_ = nullptr;
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_AOE_UTIL_H_
