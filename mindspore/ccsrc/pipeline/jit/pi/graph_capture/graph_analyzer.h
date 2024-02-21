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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_GRAPH_ANALYZER_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_GRAPH_ANALYZER_H

#include <set>
#include <vector>
#include <memory>
#include "pipeline/jit/pi/graph_capture/cfg.h"
#include "pipeline/jit/pi/graph_capture/abstract_object.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"
namespace mindspore {
namespace pijit {

class Graph;
class AbstractNode;
class ValueNode;
class CallNode;
class GraphAnalyzer;
class MindGraphAnalyzer;
using GraphAnalyzerPtr = std::shared_ptr<GraphAnalyzer>;
using MindGraphAnalyzerPtr = std::shared_ptr<MindGraphAnalyzer>;

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
  static GraphAnalyzerPtr Creator(const GraphBuilderPtr &g) {
    return g->trace_flag() ? std::static_pointer_cast<GraphAnalyzer>(std::make_shared<MindGraphAnalyzer>(g))
                           : std::make_shared<GraphAnalyzer>(g->GetGraph());
  }
  auto &GetCaptureInfo() { return info_; }
  const auto &GetCaptureInfo() const { return info_; }
  virtual void Analyze();
  bool HasTensorOperation() const;
  virtual bool NeedInterpret() const { return need_interpret_; }

 protected:
  void AddToEscaped(ValueNode *value);
  // UD analyze
  virtual void UseDefAnalyze();
  std::vector<ValueNode *> GetAliveLocals(Graph *g);
  virtual bool AnalyzeAliveLocals(std::vector<ValueNode *> aliveNodes);
  virtual void CollectInputs();
  bool need_interpret_;
  Graph *graph_;
  CapturedInfo info_;

 private:
  bool AnalyzeRecursive(Graph *g);
  bool AnalyzeCall(CallNode *);
  bool TryToCapture(AbstractNode *value);
  bool AddToCaptured(ValueNode *value);
  bool HandleCallableToGraph(AObject *f);
  bool ProduceInterpretValue(ValueNode *v);
  void CleanCapturedValue();
  void ClearCapturedInfo();
};

class MindGraphAnalyzer : public GraphAnalyzer {
 public:
  explicit MindGraphAnalyzer(const GraphBuilderPtr &g) : GraphAnalyzer(g->GetGraph()), graph_builder_(g) {}
  void Analyze() override;

 protected:
  // UD analyze
  void UseDefAnalyze() override;
  void CollectInputs() override;
  void UpdateCapturedOrder();
  bool AnalyzeAliveLocals(std::vector<ValueNode *> aliveNodes) override;
  GraphBuilderPtr graph_builder_ = nullptr;
};

bool ValidateGraphParameters(ValueNode *i);

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_GRAPH_ANALYZER_H
