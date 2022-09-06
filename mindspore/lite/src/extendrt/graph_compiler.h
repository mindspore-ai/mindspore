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
#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_H_
#define MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_H_
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/status.h"
#include "include/common/utils/utils.h"
#include "ir/func_graph.h"
namespace mindspore {
namespace infer {
struct CompileOptions {
  int optimize_level_;
};
struct CompileResult {
  GraphId root_;
  std::vector<GraphId> control_nodes_;
  std::vector<KernelGraphPtr> graphs_;
};
struct KernelInfo {
  KernelPtr kernel_;
  bool isSubgraphKernel;
  std::vector<int64_t> inputs_;
  std::vector<int64_t> outputs_;
};
struct ExcutionPlan {
  std::vector<KernelInfo> kernels_;
  std::vector<int64_t> inputs_;
  std::vector<int64_t> outputs_;
};
class GraphCompiler : public std::enable_shared_from_this<GraphCompiler> {
 public:
  explicit GraphCompiler(const CompileOptions &opts);
  virtual ~GraphCompiler();
  ExcutionPlan Compile(FuncGraphPtr graph);

 protected:
  ExcutionPlan Schedule(const CompileResult &);
  GraphId CompileSegment(const GraphSegmentPtr &segment);
  CompileResult LinkSegment();

 protected:
  GraphPartition graph_partition_;
  GraphScheduler scheduler_;
  CompileOptions options_;
  mindspore::HashMap<GraphId, std::shared_ptr<KernelGraph>> graphs_;
};
}  // namespace infer
}  // namespace mindspore
#endif
