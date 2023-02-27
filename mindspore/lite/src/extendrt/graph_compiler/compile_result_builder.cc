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

#include "src/extendrt/graph_compiler/compile_result_builder.h"
#include <algorithm>
#include "src/extendrt/graph_compiler/anfnode_tensor_adapter.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"

using AbstractBasePtr = mindspore::abstract::AbstractBasePtr;
using AbstractTensorPtr = mindspore::abstract::AbstractTensorPtr;
using AbstractSequencePtr = mindspore::abstract::AbstractSequencePtr;

namespace mindspore {
namespace infer {
StatusCode CompileResultBuilder::BuildInputs(const AnfNodePtrList &inputs) {
  MS_ASSERT(graph_ != nullptr);
  if (graph_->InputSize() > 0) {
    MS_LOG(ERROR) << "Please don't call BuildOutputs twice.";
    return kLiteError;
  }
  std::vector<Tensor *> results;
  for (auto &input : inputs) {
    results.clear();
    auto ret = CreateTensorsFromAbstract(input->abstract(), &results);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Create tensors from abstract of segments input failed, input : "
                    << input->fullname_with_scope();
      return ret;
    }
    auto arg_node = new CompileNode(input->fullname_with_scope());
    ret = graph_->AppendArgNode(arg_node);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Append input lite-node to graph failed, input : " << input->fullname_with_scope();
      return ret;
    }
    for (auto &result : results) {
      arg_node->AppendOutputTensor(result);
      ret = graph_->AppendInputTensor(result);
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "Append output tensor to argument node failed, node: " << input->fullname_with_scope();
        return ret;
      }
    }
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::BuildNodes(const GraphSegmentPtr &graph_segment) {
  MS_ASSERT(graph_ != nullptr);
  if (graph_->NodeSize() > 0) {
    MS_LOG(ERROR) << "Please don't call BuildNodes twice.";
    return kLiteError;
  }
  for (auto &node : graph_segment->nodes_) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto ret = CreateAndAppendNode(utils::cast<CNodePtr>(node));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Create compile node from cnode failed : " << node;
      return ret;
    }
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::BuildOutputs(const AnfNodePtrList &outputs) {
  MS_ASSERT(graph_ != nullptr);
  if (graph_->OutputSize() > 0) {
    MS_LOG(ERROR) << "Please don't call BuildOutputs twice.";
    return kLiteError;
  }
  for (auto &output : outputs) {
    auto out_cnode = utils::cast<CNodePtr>(output);
    if (out_cnode == nullptr) {
      MS_LOG(ERROR) << "Outputs should be a CNode vector, but got " << output->Type() << " type element.";
      return kLiteError;
    }
    for (auto &input : out_cnode->inputs()) {
      if (!utils::isa<CNodePtr>(input)) {
        continue;
      }
      auto compile_node = graph_->GetNode(input->fullname_with_scope());
      if (compile_node == nullptr) {
        continue;
      }
      for (auto &tensor : compile_node->GetOutputs()) {
        auto ret = graph_->AppendOutputTensor(tensor, true);
        if (ret != kSuccess) {
          MS_LOG(ERROR) << "Append output tensor to graph failed, output: " << out_cnode->fullname_with_scope();
          return ret;
        }
      }
    }
  }
  return kSuccess;
}

void CompileResultBuilder::IsolateTensor(Tensor *dst_tensor, const CompileNode *node, size_t index) {
  if (node == nullptr || index >= node->OutputSize()) {
    return;
  }
  auto *src_tensor = node->GetOutput(index);
  // used as inputs of other node
  auto &nodes = graph_->GetMutableNodes();
  for (auto &compile_node : nodes) {
    if (compile_node == nullptr) {
      continue;
    }
    compile_node->ReplaceInputTensor(dst_tensor, src_tensor);
  }
  // used as outputs of graph
  auto &inputs = graph_->GetMutableInputs();
  std::replace_if(
    inputs.begin(), inputs.end(), [&src_tensor](Tensor *ele) { return ele == src_tensor; }, dst_tensor);
}

StatusCode CompileResultBuilder::RemoveMakeSeqNode() {
  auto &nodes = graph_->GetMutableNodes();
  for (auto iter = nodes.begin(); iter != nodes.end();) {
    auto &node = *iter;
    if (node->GetType() != prim::kMakeTuple && node->GetType() != prim::kMakeList) {
      iter++;
      continue;
    }
    MS_LOG(INFO) << "Handling make sequence node: " << node->GetName();
    auto tensor_number = node->InputSize();
    if (tensor_number != node->OutputSize()) {
      MS_LOG(ERROR) << "MakeSequence node should has same number of inputs and outputs, but got " << tensor_number
                    << " inputs and " << node->OutputSize() << " outputs.";
      return kLiteError;
    }
    for (size_t i = 0; i < tensor_number; i++) {
      IsolateTensor(node->GetInput(i), node, i);
    }
    iter = nodes.erase(iter);
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::RemoveDependNode() {
  auto &nodes = graph_->GetMutableNodes();
  for (auto iter = nodes.begin(); iter != nodes.end();) {
    auto &node = *iter;
    if (node->GetType() != prim::kDepend) {
      iter++;
      continue;
    }
    MS_LOG(INFO) << "Handling Depend node: " << node->GetName();
    constexpr int kSize2 = 2;
    if (node->InputSize() != kSize2) {
      MS_LOG(ERROR) << "Depend node should has 2 inputs, but got " << node->InputSize();
      return kLiteError;
    }
    if (node->OutputSize() != 1) {
      MS_LOG(ERROR) << "Depend node should has 1 outputs, but got " << node->OutputSize();
      return kLiteError;
    }
    IsolateTensor(node->GetInput(0), node, 0);
    iter = nodes.erase(iter);
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::RemoveSeqGetItemNode() {
  auto &nodes = graph_->GetMutableNodes();
  for (auto iter = nodes.begin(); iter != nodes.end();) {
    auto &node = *iter;
    if (node->GetType() != prim::kTupleGetItem && node->GetType() != prim::kListGetItem &&
        node->GetType() != "array_getitem" && node->GetType() != prim::kSliceGetItem) {
      iter++;
      continue;
    }
    MS_LOG(INFO) << "Handling GetItem node: " << node->GetName();
    constexpr int kSize2 = 2;
    if (node->InputSize() != kSize2) {
      MS_LOG(ERROR) << "GetItem node should has 2 inputs, but got " << node->InputSize();
      return kLiteError;
    }
    if (node->OutputSize() != 1) {
      MS_LOG(ERROR) << "GetItem node should has 1 outputs, but got " << node->OutputSize();
      return kLiteError;
    }
    auto index_tensor = node->GetInput(1);
    if (index_tensor->data() == nullptr) {
      MS_LOG(ERROR) << "`index_tensor` of GetItem should be a const tensor, but has no data.";
      return kLiteError;
    }
    if (index_tensor->data_type() == kNumberTypeInt32) {
      auto idx = reinterpret_cast<int32_t *>(index_tensor->data())[0];
      IsolateTensor(node->GetInput(idx), node, 0);
    } else if (index_tensor->data_type() == kNumberTypeInt64) {
      auto idx = reinterpret_cast<int64_t *>(index_tensor->data())[0];
      IsolateTensor(node->GetInput(idx), node, 0);
    } else {
      MS_LOG(ERROR) << "`index_tensor` of GetItem should be a const tensor with int data type, but got "
                    << index_tensor->data_type();
      return kLiteError;
    }
    iter = nodes.erase(iter);
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::OptimizeGraph() {
  MS_ASSERT(graph_ != nullptr);
  auto ret = RemoveDependNode();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Handle Depend node failed";
    return ret;
  }
  ret = RemoveMakeSeqNode();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Handle Make Sequence node failed";
    return ret;
  }
  ret = RemoveSeqGetItemNode();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Handle Sequence-Getitem node failed";
    return ret;
  }
  return kSuccess;
}

CompileResultPtr CompileResultBuilder::Build(const GraphSegmentPtr &graph_segment, const AnfNodePtrList &inputs,
                                             const AnfNodePtrList &outputs) {
  graph_ = std::make_shared<CompileResult>(graph_format_);
  if (BuildInputs(inputs) != kSuccess) {
    MS_LOG(ERROR) << "Build graph inputs failed";
    return nullptr;
  }
  if (BuildNodes(graph_segment) != kSuccess) {
    MS_LOG(ERROR) << "Build graph nodes failed";
    return nullptr;
  }
  if (BuildOutputs(outputs) != kSuccess) {
    MS_LOG(ERROR) << "Build graph outputs failed";
    return nullptr;
  }
  if (OptimizeGraph() != kSuccess) {
    MS_LOG(ERROR) << "Optimize graph failed";
    return nullptr;
  }
  graph_->Assemble();
  return graph_;
}

StatusCode CompileResultBuilder::AppendInputCNodeToInputs(const CNodePtr &cnode, const CompileNode *compile_node) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "Input cnode is nullptr.";
    return kLiteInputParamInvalid;
  }
  if (compile_node == nullptr) {
    MS_LOG(ERROR) << "Input compile_node is nullptr.";
    return kLiteInputParamInvalid;
  }
  auto input_node = graph_->GetNode(cnode->fullname_with_scope());
  if (input_node == nullptr) {
    input_node = graph_->GetArgNode(cnode->fullname_with_scope());
  }
  if (input_node == nullptr) {
    MS_LOG(ERROR) << "Can not find input lite-node in graph, node: " << cnode->fullname_with_scope();
    return kLiteError;
  }
  for (auto &input_node_output : input_node->GetOutputs()) {
    auto ret = graph_->AppendNodeInputTensor(compile_node, input_node_output, true);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Append input tensor for node failed, node: " << compile_node->GetName();
      return ret;
    }
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::AppendInputParameterToInputs(const ParameterPtr &param_node,
                                                              const CompileNode *compile_node) {
  if (param_node == nullptr) {
    MS_LOG(ERROR) << "Input param_node is nullptr.";
    return kLiteInputParamInvalid;
  }
  if (compile_node == nullptr) {
    MS_LOG(ERROR) << "Input compile_node is nullptr.";
    return kLiteInputParamInvalid;
  }
  auto arg_node = graph_->GetArgNode(param_node->fullname_with_scope());
  if (arg_node != nullptr) {
    for (auto &output : arg_node->GetOutputs()) {
      auto ret = graph_->AppendNodeInputTensor(compile_node, output, true);
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "Append input tensor for node failed, node: " << compile_node->GetName();
        return ret;
      }
    }
    return kSuccess;
  }
  auto tensor = TensorAdapter::Convert2Tensor(param_node);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Create tensor from Parameter failed.";
    return kLiteError;
  }
  auto format_value = compile_node->GetBaseOperator()->GetAttr(mindspore::ops::kFormat);
  if (format_value != nullptr) {
    tensor->set_format(static_cast<Format>(GetValue<int64_t>(format_value)));
  } else {
    tensor->set_format(graph_format_);
  }
  auto ret = graph_->AppendNodeInputTensor(compile_node, tensor);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Append input tensor for node failed, node: " << compile_node->GetName();
    return ret;
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::AppendInputValueNodeToInputs(const ValueNodePtr &value_node,
                                                              const CompileNode *compile_node) {
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "Input value_node is nullptr.";
    return kLiteInputParamInvalid;
  }
  if (compile_node == nullptr) {
    MS_LOG(ERROR) << "Input compile_node is nullptr.";
    return kLiteInputParamInvalid;
  }
  auto tensor = TensorAdapter::Convert2Tensor(value_node);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Create tensor from ValueNode failed.";
    return kLiteError;
  }
  auto format_value = compile_node->GetBaseOperator()->GetAttr(mindspore::ops::kFormat);
  if (format_value != nullptr) {
    tensor->set_format(static_cast<Format>(GetValue<int64_t>(format_value)));
  } else {
    tensor->set_format(graph_format_);
  }
  auto ret = graph_->AppendNodeInputTensor(compile_node, tensor);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Append input tensor for node failed, node: " << compile_node->GetName();
    return ret;
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::CreateAndAppendNode(const CNodePtr &cnode) {
  auto compile_node = CompileNode::Create(cnode);
  if (compile_node == nullptr) {
    MS_LOG(ERROR) << "Create compile node failed, cnode: " << cnode->fullname_with_scope();
    return kLiteError;
  }
  auto ret = graph_->AppendNode(compile_node);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Append compile_node to graph failed, node: " << compile_node->GetName();
    return ret;
  }
  // inputs
  for (size_t i = 1; i < cnode->size(); i++) {
    auto &input = cnode->input(i);
    if (utils::isa<CNodePtr>(input)) {
      ret = this->AppendInputCNodeToInputs(utils::cast<CNodePtr>(input), compile_node);
    } else if (utils::isa<Parameter>(input)) {
      ret = this->AppendInputParameterToInputs(utils::cast<ParameterPtr>(input), compile_node);
    } else if (utils::isa<ValueNode>(input)) {
      ret = this->AppendInputValueNodeToInputs(utils::cast<ValueNodePtr>(input), compile_node);
    } else {
      MS_LOG(ERROR) << "Unsupported input node of cnode: " << input
                    << ", current cnode: " << cnode->fullname_with_scope();
      ret = kLiteNotSupport;
    }
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Create input tensor for cnode failed, cnode: " << cnode->fullname_with_scope();
      return ret;
    }
  }
  // outputs
  ret = BuildNodeOutputTensor(cnode, compile_node);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Create output tensors of cnode failed, cnode: " << cnode;
    return ret;
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::CreateTensorsFromAbstract(const AbstractBasePtr &abstract,
                                                           std::vector<Tensor *> *results) {
  if (results == nullptr) {
    MS_LOG(ERROR) << "Result is nullptr.";
    return kLiteInputParamInvalid;
  }
  results->clear();
  // multi output abstract
  if (utils::isa<AbstractSequencePtr>(abstract)) {
    auto elements = utils::cast<AbstractSequencePtr>(abstract)->elements();
    for (auto &element : elements) {
      auto tensor = TensorAdapter::Convert2Tensor(element);
      if (tensor == nullptr) {
        MS_LOG(ERROR) << "Create tensor from abstract failed, abstract : " << element;
        return kLiteError;
      }
      results->emplace_back(tensor);
    }
    return kSuccess;
  }
  // single output abstract
  if (utils::isa<AbstractTensorPtr>(abstract)) {
    auto tensor = TensorAdapter::Convert2Tensor(abstract);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor from abstract failed, abstract : " << abstract;
      return kLiteError;
    }
    results->emplace_back(tensor);
    return kSuccess;
  }
  MS_LOG(ERROR) << "Unsupported abstract: " << abstract;
  return kLiteNotSupport;
}

StatusCode CompileResultBuilder::BuildNodeOutputTensor(const CNodePtr &cnode, const CompileNode *compile_node) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "Input cnode is nullptr.";
    return kLiteInputParamInvalid;
  }
  if (compile_node == nullptr) {
    MS_LOG(ERROR) << "Input compile_node is nullptr.";
    return kLiteInputParamInvalid;
  }
  std::vector<Tensor *> results;
  auto ret = CreateTensorsFromAbstract(cnode->abstract(), &results);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Create tensors from output abstract of cnode failed, cnode : " << cnode->fullname_with_scope();
    return ret;
  }
  if (compile_node->OutputSize() > 0) {
    MS_LOG(ERROR) << "Build node output twice, node : " << compile_node->GetName();
    return kLiteError;
  }
  for (auto &result : results) {
    ret = graph_->AppendNodeOutputTensor(compile_node, result);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Append output tensor to node failed, node: " << compile_node->GetName();
      return ret;
    }
  }
  return kSuccess;
}
}  // namespace infer
}  // namespace mindspore
