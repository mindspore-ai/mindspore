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
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "src/extendrt/graph_compiler/anfnode_tensor_adapter.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "src/extendrt/utils/func_graph_utils.h"

using AbstractBasePtr = mindspore::abstract::AbstractBasePtr;
using AbstractTensorPtr = mindspore::abstract::AbstractTensorPtr;
using AbstractSequencePtr = mindspore::abstract::AbstractSequencePtr;

namespace mindspore {
namespace lite {
StatusCode CompileResultBuilder::BuildInputs(const AnfNodePtrList &inputs) {
  MS_ASSERT(graph_ != nullptr);
  if (graph_->InputSize() > 0) {
    MS_LOG(ERROR) << "Please don't call BuildInputs twice.";
    return kLiteError;
  }
  for (auto &input : inputs) {
    auto results = TensorAdapter::CreateTensorsFromAbstract(input->abstract(), compile_option_->graph_input_format);
    if (results.empty()) {
      MS_LOG(ERROR) << "Create tensors from abstract of segments input failed, input : "
                    << input->fullname_with_scope();
      return kLiteError;
    }
    auto arg_node = std::make_shared<CompileNode>(input->fullname_with_scope(), kernel::PrimitiveType());
    auto ret = graph_->AppendArgNode(arg_node);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Append input lite-node to graph failed, input : " << input->fullname_with_scope();
      return ret;
    }
    for (auto &result : results) {
      auto tensor = result.release();
      arg_node->AppendOutputTensor(tensor);
      ret = graph_->AppendInputTensor(tensor);
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "Append output tensor to argument node failed, node: " << input->fullname_with_scope();
        delete (tensor);
        return ret;
      }
    }
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::BuildNodes(const std::vector<AnfNodePtr> &nodes) {
  MS_ASSERT(graph_ != nullptr);
  if (graph_->NodeSize() > 0) {
    MS_LOG(ERROR) << "Please don't call BuildNodes twice.";
    return kLiteError;
  }

  for (auto &node : nodes) {
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

StatusCode CompileResultBuilder::BuildNodes(const GraphSegmentPtr &graph_segment) {
  return BuildNodes(graph_segment->nodes_);
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
    auto compile_node = graph_->GetNode(out_cnode->fullname_with_scope());
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
  return kSuccess;
}

// Replace `dst_tensor` with `src_tensor`.
void CompileResultBuilder::ReplaceTensor(InferTensor *dst_tensor, const InferTensor *src_tensor) {
  // used as inputs of other node
  auto &nodes = graph_->GetMutableNodes();
  for (auto &compile_node : nodes) {
    if (compile_node == nullptr) {
      continue;
    }
    compile_node->ReplaceInputTensor(dst_tensor, src_tensor);
  }
  // used as outputs of graph
  auto &outputs = graph_->GetMutableOutputs();
  std::replace_if(
    outputs.begin(), outputs.end(), [&src_tensor](InferTensor *ele) { return ele == src_tensor; }, dst_tensor);
}

StatusCode CompileResultBuilder::RemoveMakeSeqNode() {
  auto &nodes = graph_->GetMutableNodes();
  for (auto iter = nodes.begin(); iter != nodes.end();) {
    auto &node = *iter;
    if (node->GetType() != kMakeTupleOpName && node->GetType() != kMakeListOpName) {
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
      ReplaceTensor(node->GetInput(i), node->GetOutput(i));
    }
    iter = nodes.erase(iter);
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::RemoveDependNode() {
  auto &nodes = graph_->GetMutableNodes();
  for (auto iter = nodes.begin(); iter != nodes.end();) {
    auto &node = *iter;
    if (node->GetType() != kDependOpName) {
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
    ReplaceTensor(node->GetInput(0), node->GetOutput(0));
    iter = nodes.erase(iter);
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::RemoveSeqGetItemNode() {
  auto &nodes = graph_->GetMutableNodes();
  for (auto iter = nodes.begin(); iter != nodes.end();) {
    auto &node = *iter;
    if (node->GetType() != kTupleGetItemOpName && node->GetType() != kListGetItemOpName &&
        node->GetType() != "array_getitem" && node->GetType() != kSliceGetItemOpName) {
      iter++;
      continue;
    }
    MS_LOG(DEBUG) << "Handling GetItem node: " << node->GetName();
    if (node->OutputSize() != 1) {
      MS_LOG(ERROR) << "GetItem node should has 1 outputs, but got " << node->OutputSize();
      return kLiteError;
    }
    auto index_tensor = node->GetInput(node->GetInputs().size() - 1);
    if (index_tensor->data() == nullptr) {
      MS_LOG(ERROR) << "`index_tensor` of GetItem should be a const tensor, but has no data.";
      return kLiteError;
    }
    if (index_tensor->data_type() == kNumberTypeInt32) {
      auto idx = reinterpret_cast<int32_t *>(index_tensor->data())[0];
      ReplaceTensor(node->GetInput(idx), node->GetOutput(0));
    } else if (index_tensor->data_type() == kNumberTypeInt64) {
      auto idx = reinterpret_cast<int64_t *>(index_tensor->data())[0];
      ReplaceTensor(node->GetInput(idx), node->GetOutput(0));
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
  graph_ = std::make_shared<CompileResult>();
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

StatusCode CompileResultBuilder::AppendInputCNodeToInputs(const CNodePtr &cnode, const CompileNodePtr &compile_node) {
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
                                                              const CompileNodePtr &compile_node) {
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
  auto tensor_from_param = TensorAdapter::Convert2Tensor(param_node);
  if (tensor_from_param == nullptr) {
    MS_LOG(ERROR) << "Create tensor from Parameter failed.";
    return kLiteError;
  }
  auto format_value = compile_node->GetBaseOperator()->GetAttr(mindspore::ops::kFormat);
  if (format_value != nullptr) {
    tensor_from_param->set_format(static_cast<Format>(GetValue<int64_t>(format_value)));
  } else {
    tensor_from_param->set_format(compile_option_->graph_format);
  }
  auto ret = graph_->AppendNodeInputTensor(compile_node, tensor_from_param);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Append input tensor for node failed, node: " << compile_node->GetName();
    delete tensor_from_param;
    return ret;
  }
  return kSuccess;
}

StatusCode CompileResultBuilder::AppendInputValueNodeToInputs(const ValueNodePtr &value_node,
                                                              const CompileNodePtr &compile_node) {
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "Input value_node is nullptr.";
    return kLiteInputParamInvalid;
  }
  if (compile_node == nullptr) {
    MS_LOG(ERROR) << "Input compile_node is nullptr.";
    return kLiteInputParamInvalid;
  }
  if (value_node->value() != nullptr && value_node->value()->isa<Monad>()) {
    MS_LOG(WARNING) << "Skip Monad value node: " << value_node->fullname_with_scope();
    return kSuccess;
  }
  auto tensor_from_value = TensorAdapter::Convert2Tensor(value_node);
  if (tensor_from_value == nullptr) {
    MS_LOG(ERROR) << "Create tensor from ValueNode failed.";
    return kLiteError;
  }
  auto format_value = compile_node->GetBaseOperator()->GetAttr(mindspore::ops::kFormat);
  if (format_value != nullptr) {
    tensor_from_value->set_format(static_cast<Format>(GetValue<int64_t>(format_value)));
  } else {
    tensor_from_value->set_format(compile_option_->graph_format);
  }
  auto ret = graph_->AppendNodeInputTensor(compile_node, tensor_from_value);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Append input tensor for node failed, node: " << compile_node->GetName();
    delete tensor_from_value;
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

StatusCode CompileResultBuilder::BuildNodeOutputTensor(const CNodePtr &cnode, const CompileNodePtr &compile_node) {
  if (compile_node == nullptr) {
    MS_LOG(ERROR) << "Input compile_node is nullptr.";
    return kLiteInputParamInvalid;
  }
  if (compile_node->OutputSize() > 0) {
    MS_LOG(ERROR) << "Build node output twice, node : " << compile_node->GetName();
    return kLiteError;
  }
  auto results = TensorAdapter::Convert2Tensor(cnode);
  if (results.empty()) {
    MS_LOG(ERROR) << "Create tensors from cnode failed, cnode : " << cnode->fullname_with_scope();
    return kLiteError;
  }
  size_t index = 0;
  auto ret = kSuccess;
  for (; index < results.size(); index++) {
    auto tensor = results[index];
    ret = graph_->AppendNodeOutputTensor(compile_node, tensor);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Append output tensor to node failed, node: " << compile_node->GetName();
      break;
    }
  }
  // release results if failed
  for (; index < results.size(); index++) {
    delete results[index];
  }
  return ret;
}

StatusCode CompileResultBuilder::BuildNodes(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto nodes = func_graph->TopoSort(func_graph->get_return());
  if (nodes.empty()) {
    MS_LOG(ERROR) << "There are no nodes in the graph";
    return kLiteError;
  }

  return BuildNodes(nodes);
}

CompileResultPtr CompileResultBuilder::Build(const FuncGraphPtr &func_graph) {
  graph_ = std::make_shared<CompileResult>();

  if (BuildInputs(func_graph->get_inputs()) != kSuccess) {
    MS_LOG(ERROR) << "Build graph inputs failed";
    return nullptr;
  }
  if (BuildNodes(func_graph) != kSuccess) {
    MS_LOG(ERROR) << "Build graph nodes failed";
    return nullptr;
  }

  std::vector<AnfWithOutIndex> outputs_with_index;
  FuncGraphUtils::GetFuncGraphOutputs(func_graph, &outputs_with_index);
  AnfNodePtrList outputs;
  outputs.resize(outputs_with_index.size());
  for (auto &output : outputs_with_index) {
    if (output.second >= outputs.size()) {
      MS_LOG(ERROR) << "Build graph nodes failed";
      return nullptr;
    }
    outputs[output.second] = output.first;
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
}  // namespace lite
}  // namespace mindspore
