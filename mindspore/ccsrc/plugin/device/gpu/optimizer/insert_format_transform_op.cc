/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/optimizer/insert_format_transform_op.h"
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"

namespace mindspore {
namespace opt {
namespace {
const int kFakeTransposeShapeOneNum = 2;

AnfNodePtr ConvertValueToTensor(const KernelGraphPtr &kernel_graph, const ValueNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  tensor::TensorPtr tensor_ptr = CreateTupleTensor(value->cast<ValueTuplePtr>());
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  auto tensor_input = std::make_shared<ValueNode>(tensor_ptr);
  MS_EXCEPTION_IF_NULL(tensor_input);
  tensor_input->set_abstract(tensor_ptr->ToAbstract());
  tensor_input = kernel_graph->NewValueNode(tensor_input);
  kernel_graph->AddValueNodeToGraph(tensor_input);
  tensor_input->set_scope(input_node->scope());
  return tensor_input;
}

std::vector<int64_t> TransposeAxis(const std::string &src_format, const std::string &dst_format) {
  if ((src_format == kOpFormat_NCHW) && (dst_format == kOpFormat_NHWC)) {
    return {0, 2, 3, 1};
  } else if ((src_format == kOpFormat_NHWC) && (dst_format == kOpFormat_NCHW)) {
    return {0, 3, 1, 2};
  } else {
    MS_LOG(EXCEPTION) << "Invalid format transform, from " << src_format << " to " << dst_format;
  }
}

// Transpose can be replaceed by nop reshape in some situations.
// 1. out_shape [x, 1, 1, y]
// 2. out_shape [x, y, 1, 1]
// 3. out_shape [x, 1, y, 1]
bool IsFakeTranspose(const std::vector<int64_t> &out_shape, const std::vector<int64_t> &transpose_perm) {
  if (out_shape.size() != device::gpu::kFormatTransformDimension) {
    MS_LOG(EXCEPTION) << "Invalid data shape, 4-D data was needed, but get " << out_shape.size() << "-D.";
  }
  std::vector<int64_t> perm1 = {0, 2, 3, 1};
  std::vector<int64_t> perm2 = {0, 3, 1, 2};
  auto num = std::count(out_shape.begin(), out_shape.end(), 1);
  if ((transpose_perm == perm1) || (transpose_perm == perm2)) {
    if (num >= kFakeTransposeShapeOneNum) {
      return true;
    }
  }
  return false;
}

void SetTransposeOpBuildInfo(const std::string &input_format, const std::string &output_format,
                             const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto input_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  auto output_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({input_format, kOpFormat_DEFAULT});
  builder.SetInputsDeviceType({input_type, kNumberTypeInt64});
  builder.SetOutputsFormat({output_format});
  builder.SetOutputsDeviceType({output_type});
  builder.SetKernelType(UNKNOWN_KERNEL_TYPE);
  builder.SetProcessor(kernel::Processor::CUDA);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
}

// Insert transpose op between node and used_node whose position is used_node_index.
CNodePtr InsertTransposeOp(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &used_node,
                           int used_node_index, const std::vector<int64_t> &transpose_perm) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(used_node);
  auto kernel_graph = graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  MS_LOG(DEBUG) << "Node: " << node->fullname_with_scope() << ", used node: " << used_node->fullname_with_scope()
                << ", index: " << used_node_index;
  // 0.Judge whether it is a fake transpose
  auto transed_shape = AnfAlgo::GetInputDeviceShape(used_node, used_node_index);
  bool is_fake = IsFakeTranspose(transed_shape, transpose_perm);
  // 1.Create a transpose node or a fake transpose node:reshape.
  mindspore::PrimitivePtr transpose_prim;
  if (is_fake) {
    transpose_prim = std::make_shared<Primitive>(prim::kPrimReshape->name());
  } else {
    transpose_prim = std::make_shared<Primitive>(prim::kPrimTranspose->name());
  }
  MS_EXCEPTION_IF_NULL(transpose_prim);
  // 2.Set the input of transpose.
  std::vector<AnfNodePtr> transpose_input = {NewValueNode(transpose_prim), node};
  if (is_fake) {
    auto reshape_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(used_node, used_node_index);
    auto shape_node = NewValueNode(MakeValue<std::vector<int64_t>>(reshape_shape));
    auto shape_tensor = ConvertValueToTensor(kernel_graph, shape_node);
    transpose_input.push_back(shape_tensor);
  } else {
    auto perm_node = NewValueNode(MakeValue<std::vector<int64_t>>(transpose_perm));
    auto perm_tensor = ConvertValueToTensor(kernel_graph, perm_node);
    transpose_input.push_back(perm_tensor);
  }
  auto transpose_op = graph->NewCNode(transpose_input);
  MS_EXCEPTION_IF_NULL(transpose_op);
  // 3.Set the output info of transpose.
  auto transpose_type = {common::AnfAlgo::GetPrevNodeOutputInferDataType(used_node, used_node_index)};
  auto base_shape = AnfAlgo::GetPrevNodeOutputDetailShape(used_node, used_node_index);
  common::AnfAlgo::SetOutputTypeAndDetailShape(transpose_type, {base_shape}, transpose_op.get());

  // 4. Set the new edge of transpose op.
  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->SetEdge(used_node, used_node_index + 1, transpose_op);
  return transpose_op;
}
}  // namespace

const AnfNodePtr InsertFormatTransformOp::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  if (!AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  auto iter = device::gpu::kKernelFormatPositionMap.find(common::AnfAlgo::GetCNodeName(node));
  if (iter == device::gpu::kKernelFormatPositionMap.end()) {
    return nullptr;
  }
  auto origin_data_format = AnfAlgo::GetOriginDataFormat(node);
  if (origin_data_format == kOpFormat_DEFAULT) {
    origin_data_format = kOpFormat_NCHW;
  }
  MS_LOG(DEBUG) << "Process node: " << node->fullname_with_scope();
  // Insert input transpose from origin_data_format to input_format.
  auto inputs_format = AnfAlgo::GetAllInputFormats(node);
  for (size_t i = 0; i < inputs_format.size(); i++) {
    if ((inputs_format[i] != kOpFormat_DEFAULT) && (inputs_format[i] != origin_data_format)) {
      auto input_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), i);
      MS_EXCEPTION_IF_NULL(input_node);
      auto input_transpose_perm = TransposeAxis(origin_data_format, inputs_format[i]);
      auto input_transpose_op = InsertTransposeOp(graph, input_node, node, i, input_transpose_perm);
      SetTransposeOpBuildInfo(kOpFormat_DEFAULT, inputs_format[i], input_transpose_op);
    }
  }

  // Insert output transpose from output_format to origin_data_format.
  auto outputs_format = AnfAlgo::GetAllOutputFormats(node);
  for (size_t i = 0; i < outputs_format.size(); i++) {
    if ((outputs_format[i] != kOpFormat_DEFAULT) && (outputs_format[i] != origin_data_format)) {
      // Find all nodes connected with node output, and change their inputs to transpose.
      auto used_node_list = GetRealNodeUsedListByOutputIdx(graph, node, i);
      MS_EXCEPTION_IF_NULL(used_node_list);
      for (size_t j = 0; j < used_node_list->size(); j++) {
        auto used_node = used_node_list->at(j).first;
        auto used_node_index = used_node_list->at(j).second - 1;
        auto output_transpose_perm = TransposeAxis(outputs_format[i], origin_data_format);
        if (common::AnfAlgo::GetCNodeName(used_node) == prim::kPrimTupleGetItem->name()) {
          MS_LOG(DEBUG) << "The used node of [" << node->fullname_with_scope() << "] is tuple item.";
          // The tuple item need get next used nodes again.
          ProcessForTupleItem(graph, used_node, used_node_index, output_transpose_perm, outputs_format[i]);
          continue;
        }
        auto output_transpose_op = InsertTransposeOp(graph, node, used_node, used_node_index, output_transpose_perm);
        SetTransposeOpBuildInfo(outputs_format[i], kOpFormat_DEFAULT, output_transpose_op);
      }
    }
  }
  return node;
}

void InsertFormatTransformOp::ProcessForTupleItem(const FuncGraphPtr &graph, const AnfNodePtr &node, int node_index,
                                                  const std::vector<int64_t> &transpose_perm,
                                                  const std::string &transpose_format) const {
  auto used_node_list = GetRealNodeUsedListByOutputIdx(graph, node, node_index);
  MS_EXCEPTION_IF_NULL(used_node_list);
  for (size_t i = 0; i < used_node_list->size(); i++) {
    auto used_node = used_node_list->at(i).first;
    auto used_node_index = used_node_list->at(i).second - 1;
    if (common::AnfAlgo::GetCNodeName(used_node) == prim::kPrimTupleGetItem->name()) {
      MS_LOG(EXCEPTION) << "The used node of tuple item can't be tuple item.";
    }
    auto transpose_op = InsertTransposeOp(graph, node, used_node, used_node_index, transpose_perm);
    SetTransposeOpBuildInfo(transpose_format, kOpFormat_DEFAULT, transpose_op);
  }
}
}  // namespace opt
}  // namespace mindspore
