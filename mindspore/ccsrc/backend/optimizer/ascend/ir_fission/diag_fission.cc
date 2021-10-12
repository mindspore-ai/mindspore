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
#include "backend/optimizer/ascend/ir_fission/diag_fission.h"
#include <algorithm>
#include <memory>
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/optimizer/opt.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kDiagInputNum = 1;
constexpr size_t kDiagInputMaxDim = 4;

template <typename T>
void SetAssistTensorData(void *data, const T &value, size_t dims_size) {
  MS_EXCEPTION_IF_NULL(data);
  auto tensor_data = reinterpret_cast<T *>(data);
  for (size_t i = 0; i < dims_size; ++i) {
    tensor_data[(1 + dims_size) * i] = value;
  }
}
}  // namespace

ValueNodePtr DiagFission::CreateAssistNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const std::vector<size_t> &ori_shape) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  std::vector<size_t> output_shape(ori_shape);
  size_t dims = 1;
  for (size_t i = 0; i < ori_shape.size(); i++) {
    dims = dims * ori_shape[i];
  }
  (void)output_shape.insert(output_shape.end(), ori_shape.begin(), ori_shape.end());
  auto type = AnfAlgo::GetOutputInferDataType(node, 0);
  std::vector<int64_t> assist_shape;
  std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(assist_shape), SizeToLong);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type, assist_shape);
  AbstractBasePtr x_abstract;
  if (type == kNumberTypeInt32) {
    SetAssistTensorData<int32_t>(tensor->data_c(), 1, dims);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, assist_shape);
  } else if (type == kNumberTypeFloat16) {
    SetAssistTensorData<float16>(tensor->data_c(), float16(static_cast<float>(1)), dims);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, assist_shape);
  } else if (type == kNumberTypeFloat32) {
    SetAssistTensorData<float>(tensor->data_c(), static_cast<float>(1), dims);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat, assist_shape);
  } else {
    MS_EXCEPTION(TypeError) << "The type of node [" << node->DebugString()
                            << "] should be int32, float16 or float32, but got" << node->Type()->ToString();
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto assist_value_node = kernel_graph->NewValueNode(x_abstract, tensor);
  kernel_graph->AddValueNodeToGraph(assist_value_node);
  AnfAlgo::SetOutputInferTypeAndShape({type}, {output_shape}, assist_value_node.get());
  return assist_value_node;
}

const BaseRef DiagFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto diag_prim = std::make_shared<Primitive>(prim::kPrimDiag->name());
  return VectorRef({diag_prim, Xs});
}

const AnfNodePtr DiagFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  auto diag_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(diag_cnode);
  if (diag_cnode->size() != kDiagInputNum + 1) {
    MS_LOG(INFO) << "The node " << diag_cnode->DebugString() << " is not equal to " << kDiagInputNum << " inputs";
    return nullptr;
  }
  auto input_shape = AnfAlgo::GetOutputInferShape(diag_cnode->inputs()[kIndex1], 0);
  if (input_shape.size() > kDiagInputMaxDim) {
    MS_EXCEPTION(ValueError) << "For Diag, rank of input should be less than 5, but got: " << input_shape.size();
  }
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimDiag->name()))};
  auto assist_const = CreateAssistNode(graph, diag_cnode, input_shape);
  (void)new_inputs.insert(new_inputs.end(), diag_cnode->inputs().begin() + 1, diag_cnode->inputs().end());
  new_inputs.push_back(assist_const);
  CNodePtr new_cnode = graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(diag_cnode->abstract());
  new_cnode->set_scope(diag_cnode->scope());
  if (kernel_graph != nullptr) {
    kernel_graph->AddValueNodeToGraph(assist_const);
    MS_LOG(INFO) << "Add assist tensor for diag op success.";
  }
  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
