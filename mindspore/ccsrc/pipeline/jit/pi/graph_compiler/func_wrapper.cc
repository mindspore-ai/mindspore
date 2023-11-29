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
#include "pipeline/jit/pi/graph_compiler/func_wrapper.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "pipeline/jit/pi/graph_compiler/pi_ir/custom_nodes.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/value.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace jit {
namespace graph {
const ValuePtrList &InputsCollector::GetInputs() {
  VISIT_NODE_LIST(nodes_)
  return inputs_;
}

void InputsCollector::Visit_(const ir::LoadValueNodePtr &node) {
  if (node->GetOpCode() == LOAD_FAST) {
    AddInput(node->GetArg(0));
  } else {
    VISIT_NODE_LIST(node->GetArgs())
  }
}

void InputsCollector::Visit_(const ir::StoreNodePtr &node) {
  Visit(node->GetArg(0));
  if (node->GetOpCode() == STORE_FAST) {
    AddAssignedVar(node->GetArg(1));
  }
}

void InputsCollector::AddInput(const ir::NodePtr &input) {
  MS_EXCEPTION_IF_CHECK_FAIL(input->isa<ir::Value>(), input->ToString() + " is not excepted.");
  auto value = input->cast<ir::ValuePtr>();
  if ((assigned_vars_.find(value->GetValue()) == assigned_vars_.end()) &&
      (input_names_.find(value->GetValue()) == input_names_.end())) {
    input_names_.insert(value->GetValue());
    inputs_.push_back(value);
  }
}

void InputsCollector::AddAssignedVar(const ir::NodePtr &var) {
  MS_EXCEPTION_IF_CHECK_FAIL(var->isa<ir::Value>(), var->ToString() + " is not excepted.");
  auto value = var->cast<ir::ValuePtr>();
  assigned_vars_.insert(value->GetValue());
}

const ValuePtrList &OutputsCollector::GetOutputs() {
  VISIT_NODE_LIST(nodes_)
  return outputs_;
}

void OutputsCollector::Visit_(const ir::StoreNodePtr &node) {
  if (node->GetOpCode() == STORE_FAST) {
    AddOutput(node->GetRightArg());
  } else {
    Visit(node->GetLeftArg());
    Visit(node->GetRightArg());
  }
}

void OutputsCollector::AddOutput(const ir::NodePtr &output) {
  MS_EXCEPTION_IF_CHECK_FAIL(output->isa<ir::Value>(), output->ToString() + " is not excepted.");
  auto value = output->cast<ir::ValuePtr>();
  if (output_names_.find(value->GetValue()) == output_names_.end()) {
    output_names_.insert(value->GetValue());
    outputs_.push_back(value);
  }
}

ir::FunctionNodePtr FuncWrapper::Wrapper() {
  (void)GetOutputs();
  GenerateReturn();
  GenerateParameters();
  return func_;
}

const ValuePtrList &FuncWrapper::GetOutputs() {
  if (outputs_.empty()) {
    auto output_collector = std::make_shared<OutputsCollector>(func_->GetNodes());
    outputs_ = output_collector->GetOutputs();
  }
  return outputs_;
}

void FuncWrapper::GenerateParameters() const {
  auto intput_collector = std::make_shared<InputsCollector>(func_->GetNodes());
  auto inputs = intput_collector->GetInputs();
  size_t index = 0;
  std::for_each(inputs.begin(), inputs.end(), [&index, this](const ir::ValuePtr &input) {
    std::string name = py::cast<std::string>(input->GetValue());
    ir::ParameterPtr param = std::make_shared<ir::Parameter>(index, name);
    // Set arg as positional parameter
    param->SetCategory(ir::Parameter::POSITIONAL);
    func_->AddParameter(param);
    index++;
  });
  func_->SetPosArgsCnt(index);
}

void FuncWrapper::GenerateReturn() const {
  auto nodes = func_->GetNodes();
  if (!nodes.empty() && nodes.back()->isa<ir::ReturnNode>()) {
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(!outputs_.empty(), "Output can not be empty.");
  ir::NodePtrList opnds;
  std::transform(outputs_.begin(), outputs_.end(), std::back_inserter(opnds),
                 [](const ir::ValuePtr &value) { return std::make_shared<ir::LoadValueNode>(LOAD_FAST, value); });
  ir::NodePtr tuple = std::make_shared<ir::BuildNode>(BUILD_TUPLE, opnds);
  ir::ReturnNodePtr ret = std::make_shared<ir::ReturnNode>(tuple);
  func_->AddNode(ret);
}

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
