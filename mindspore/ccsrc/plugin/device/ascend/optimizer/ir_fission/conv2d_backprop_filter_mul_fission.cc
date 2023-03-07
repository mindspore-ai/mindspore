/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/conv2d_backprop_filter_mul_fission.h"

#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include "backend/common/optimizer/const_input_to_attr.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "utils/ms_context.h"

namespace mindspore::opt {
namespace {
constexpr int64_t kGroupsDefaultValue = 1;

template <typename T>
void SetAssistTensorData(void *data, const T &value, int64_t dims_size) {
  MS_EXCEPTION_IF_NULL(data);
  auto tensor_data = static_cast<T *>(data);
  for (size_t i = 0; i < static_cast<size_t>(dims_size); ++i) {
    tensor_data[i] = value;
  }
}

ValueNodePtr CreateAssistNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const ShapeVector &shape,
                              int64_t matrix_size) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type, shape);
  AbstractBasePtr x_abstract;
  if (type == kNumberTypeInt32) {
    SetAssistTensorData<int32_t>(tensor->data_c(), 1, matrix_size);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, shape);
  } else if (type == kNumberTypeFloat16) {
    SetAssistTensorData<float16>(tensor->data_c(), float16(static_cast<float>(1)), matrix_size);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, shape);
  } else if (type == kNumberTypeFloat32) {
    SetAssistTensorData<float>(tensor->data_c(), static_cast<float>(1), matrix_size);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat, shape);
  } else {
    MS_EXCEPTION(TypeError) << "The type of node [" << node->DebugString()
                            << "] should be int32, float16 or float32, but got" << node->Type()->ToString();
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto assist_value_node = kernel_graph->NewValueNode(x_abstract, tensor);
  kernel_graph->AddValueNodeToGraph(assist_value_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({type}, {shape}, assist_value_node.get());
  return assist_value_node;
}
}  // namespace

const BaseRef Conv2dBackpropFilterMul::DefinePattern() const {
  VarPtr X1 = std::make_shared<Var>();
  VarPtr X2 = std::make_shared<Var>();
  auto prim = std::make_shared<Primitive>(kConv2DBackpropFilterDOpName);
  return VectorRef({prim, X1, X2});
}

const AnfNodePtr Conv2dBackpropFilterMul::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }
  if (GetBoolAttr(node, kAttrVisited)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    MS_LOG(EXCEPTION) << "Get Conv2DBackpropFilter attr(groups) failed, node: " << node->DebugString();
  }
  auto groups = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrGroup);
  // if groups not > 1, skip process
  if (groups <= kGroupsDefaultValue) {
    return nullptr;
  }
  auto shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  if (shape.size() != kDim4) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter node output ori shape is: " << shape.size();
    return nullptr;
  }
  auto filter_n = shape[kIndex0];
  auto filter_c = shape[kIndex1];
  auto filter_h = shape[kIndex2];
  auto filter_w = shape[kIndex3];
  auto matrix_size = filter_n * filter_c * filter_h * filter_w;
  if (matrix_size <= 0 || filter_n % groups != 0) {
    MS_LOG(ERROR) << "Conv2DBackpropFilter node shape value is error, matrix_size: " << matrix_size
                  << ", shape: " << shape << ", groups: " << groups;
    return nullptr;
  }
  // CreateAssitValueNode
  auto value_node = CreateAssistNode(func_graph, node, shape, matrix_size);
  value_node->set_fracz_group(groups);
  MS_LOG(INFO) << "Create assist value node success.";
  // CreateMulNode
  std::vector<AnfNodePtr> mul_inputs{NewValueNode(std::make_shared<Primitive>(kMulOpName)), node, value_node};
  CNodePtr mul_node = NewCNode(mul_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(mul_node);
  mul_node->set_abstract(cnode->abstract());
  mul_node->set_scope(cnode->scope());
  MS_LOG(INFO) << "Create mul node success.";
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  return mul_node;
}
}  // namespace mindspore::opt
