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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_GRAPH_ANALYZER_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_GRAPH_ANALYZER_H

#include <set>
#include <vector>
#include "pipeline/jit/pi/graph_capture/cfg.h"

namespace mindspore {
namespace jit {
namespace graph {

class Graph;
class AbstractNode;
class ValueNode;
class CallNode;

class GraphAnalyzer {
 public:
  // escaped_locals and captured.values do not intersect
  struct CapturedInfo {
    struct {
      std::set<ValueNode *> inputs;
      std::set<ValueNode *> values;
      std::vector<ValueNode *> order;
    } captured_locals;
    std::set<ValueNode *> escaped_locals;
    std::vector<ValueNode *> ordered_escaped_locals;
    bool has_grad_ = false;
  };

  explicit GraphAnalyzer(Graph *g) : graph_(g) {}
  auto &GetCaptureInfo() { return info_; }
  const auto &GetCaptureInfo() const { return info_; }
  void Analyze();
  bool HasTensorOperation() const;

 private:
  bool AnalyzeRecursive(Graph *g);
  bool AnalyzeBlock(Block *g);
  bool AnalyzeCall(CallNode *);
  bool TryToCapture(AbstractNode *value);
  bool AddToCaptured(ValueNode *value);
  void AddToEscaped(ValueNode *value);
  bool ProduceInterpretValue(ValueNode *v);
  void CollectInputs();
  void CleanCapturedValue();

  Graph *graph_;
  CapturedInfo info_;
};
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_GRAPH_ANALYZER_H
