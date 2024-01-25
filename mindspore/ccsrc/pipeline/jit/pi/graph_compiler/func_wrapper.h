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

#ifndef MINDSPORE_PI_JIT_FUNC_WRAPPER_H_
#define MINDSPORE_PI_JIT_FUNC_WRAPPER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "pipeline/jit/pi/graph_compiler/pi_ir/ctrl_flow.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/custom_nodes.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/ir_visitor.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/value.h"

namespace mindspore {
namespace pijit {
namespace py = pybind11;
using ValuePtrList = std::vector<ir::ValuePtr>;

class InputsCollector final : public ir::IRVisitor {
 public:
  explicit InputsCollector(const ir::NodePtrList &nodes) : nodes_(nodes) {}
  virtual ~InputsCollector() = default;
  const ValuePtrList &GetInputs();
  void Visit_(const ir::LoadValueNodePtr &node) override;
  void Visit_(const ir::StoreNodePtr &node) override;

 private:
  void AddInput(const ir::NodePtr &input);
  void AddAssignedVar(const ir::NodePtr &var);

  const ir::NodePtrList &nodes_;
  ValuePtrList inputs_;
  std::set<py::object> input_names_;
  std::set<py::object> assigned_vars_;
};

using InputsCollectorPtr = std::shared_ptr<InputsCollector>;

class OutputsCollector final : public ir::IRVisitor {
 public:
  explicit OutputsCollector(const ir::NodePtrList &nodes) : nodes_(nodes) {}
  virtual ~OutputsCollector() = default;
  const ValuePtrList &GetOutputs();
  void Visit_(const ir::StoreNodePtr &node) override;

 private:
  void AddOutput(const ir::NodePtr &output);

  const ir::NodePtrList &nodes_;
  ValuePtrList outputs_;
  std::set<py::object> output_names_;
};

using OutputsCollectorPtr = std::shared_ptr<OutputsCollector>;

// FuncInliner to convert ir graph to function graph
class FuncWrapper {
 public:
  explicit FuncWrapper(const std::string &func_name, const ir::NodePtrList &nodes)
      : func_(std::make_shared<ir::FunctionNode>(func_name, nodes)) {}
  virtual ~FuncWrapper() = default;
  ir::FunctionNodePtr Wrapper();
  const ValuePtrList &GetOutputs();
  void SpecifyOutputs(const ValuePtrList &outputs) { outputs_ = outputs; }

 private:
  void GenerateParameters() const;
  void GenerateReturn() const;

  const ir::FunctionNodePtr func_;
  ValuePtrList outputs_;
};

using FuncWrapperPtr = std::shared_ptr<FuncWrapper>;
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_FUNC_WRAPPER_H_
