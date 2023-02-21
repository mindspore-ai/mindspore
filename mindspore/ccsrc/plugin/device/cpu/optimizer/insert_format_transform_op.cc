/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/optimizer/insert_format_transform_op.h"

#include <numeric>
#include <memory>
#include <string>
#include <vector>
#include "utils/hash_set.h"
#include "kernel/kernel_build_info.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kMinDimNeedToTransform = 3;
constexpr int64_t kNumLong1 = 1;
constexpr int64_t kNumLong2 = 2;
constexpr size_t kNumSize1 = 1;
constexpr size_t kNumSize2 = 2;
enum FormatTransformDir { ChannelFirst2ChannelLast = 0, ChannelLast2ChannelFirst };

// get perm between channel-first shape and channel-last shape.
// eg. 4D channel-first => channel-last: [0,1,2,3] => [0,2,3,1];
// eg. 4D channel-last => channel-first: [0,1,2,3] => [0,3,1,2];
std::vector<int64_t> TransposeAxis(const int dim, FormatTransformDir dir) {
  std::vector<int64_t> axis;
  axis.resize(dim);
  if (dir == ChannelFirst2ChannelLast) {
    std::iota(axis.begin() + kNumSize1, axis.end(), kNumLong2);
    axis[dim - 1] = 1;
  } else {
    std::iota(axis.begin() + kNumSize2, axis.end(), kNumLong1);
    axis[1] = dim - 1;
  }
  return axis;
}

ValueNodePtr CreateValueNode(const std::vector<int64_t> &transpose_perm, const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto tensor_ptr = std::make_shared<tensor::Tensor>(transpose_perm, TypeIdToType(kNumberTypeInt64));
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  auto value_node = std::make_shared<ValueNode>(tensor_ptr);
  MS_EXCEPTION_IF_NULL(value_node);
  value_node->set_abstract(tensor_ptr->ToAbstract());

  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  if (kernel_graph != nullptr) {
    value_node = kernel_graph->NewValueNode(value_node);
    kernel_graph->AddValueNodeToGraph(value_node);
  } else {
    value_node = MakeValueNode(value_node);
  }

  return value_node;
}

CNodePtr InsertTransposeOp(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &used_node,
                           int used_node_index, const std::vector<int64_t> &transpose_perm) {
  MS_LOG(DEBUG) << "Node: " << node->fullname_with_scope() << ", used node: " << used_node->fullname_with_scope()
                << ", index: " << used_node_index;
  MS_EXCEPTION_IF_NULL(graph);
  // 1.Create a transpose node or a fake transpose node:reshape.
  auto primitive_ptr = prim::kPrimTranspose;
  auto transpose_prim = std::make_shared<Primitive>(primitive_ptr->name());
  MS_EXCEPTION_IF_NULL(transpose_prim);
  // 2.Set the input of transpose.
  auto perm_value_node = CreateValueNode(transpose_perm, graph);
  std::vector<AnfNodePtr> transpose_input = {NewValueNode(transpose_prim), node, perm_value_node};
  auto transpose_op = graph->NewCNode(transpose_input);
  // 3.Set the output info of transpose.
  auto transpose_type = {common::AnfAlgo::GetPrevNodeOutputInferDataType(used_node, IntToSize(used_node_index))};
  auto transpose_shape = {AnfAlgo::GetPrevNodeOutputDetailShape(used_node, IntToSize(used_node_index))};
  common::AnfAlgo::SetOutputTypeAndDetailShape(transpose_type, transpose_shape, transpose_op.get());
  // 4. Set the new edge of transpose op.
  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->SetEdge(used_node, used_node_index + 1, transpose_op);
  return transpose_op;
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
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
}

void ProcessForTupleItem(const FuncGraphPtr &graph, const AnfNodePtr &node, int node_index,
                         const std::vector<int64_t> &transpose_perm, const std::string &transpose_format) {
  auto used_node_list = GetRealNodeUsedListByOutputIdx(graph, node, node_index);
  for (size_t i = 0; i < used_node_list->size(); i++) {
    auto used_node = used_node_list->at(i).first;
    auto used_node_index = used_node_list->at(i).second - 1;
    if (common::AnfAlgo::GetCNodeName(used_node) == prim::kPrimTupleGetItem->name()) {
      MS_LOG(EXCEPTION) << "The used node of tuple item " << used_node->DebugString() << " can't be tuple item.";
    }

    // node->used_node, if output format of node equals input format of used_node,
    // then no need to insert transpose between node and used_node.
    auto used_node_in_format = AnfUtils::IsRealCNodeKernel(used_node)
                                 ? AnfAlgo::GetInputFormat(used_node, IntToSize(used_node_index))
                                 : kOpFormat_DEFAULT;
    if (transpose_format == used_node_in_format) {
      continue;
    }
    auto transpose_op = InsertTransposeOp(graph, node, used_node, used_node_index, transpose_perm);
    SetTransposeOpBuildInfo(transpose_format, kOpFormat_DEFAULT, transpose_op);
  }
}

void InsertTransformOpForInput(const FuncGraphPtr &graph, const AnfNodePtr &node, const std::string &origin_format) {
  auto inputs_format = AnfAlgo::GetAllInputFormats(node);
  for (size_t i = 0; i < inputs_format.size(); ++i) {
    if ((inputs_format[i] == kOpFormat_DEFAULT) || (inputs_format[i] == origin_format)) {
      continue;
    }
    auto prev_input_format = AnfAlgo::GetPrevNodeOutputFormat(node, i);
    if (inputs_format[i] == prev_input_format) {
      continue;
    }
    auto in_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, i);
    auto dim = in_shape.size();
    if (dim < kMinDimNeedToTransform) {
      continue;
    }
    auto input_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), i);
    MS_EXCEPTION_IF_NULL(input_node);
    auto transpose_perm = TransposeAxis(dim, ChannelFirst2ChannelLast);
    auto transpose_op = InsertTransposeOp(graph, input_node, node, i, transpose_perm);
    SetTransposeOpBuildInfo(kOpFormat_DEFAULT, inputs_format[i], transpose_op);
  }
}

// Insert output transpose from output_format to origin_format.
void InsertTransformOpForOutput(const FuncGraphPtr &graph, const AnfNodePtr &node, const std::string &origin_format) {
  auto outputs_format = AnfAlgo::GetAllOutputFormats(node);
  for (size_t i = 0; i < outputs_format.size(); ++i) {
    if ((outputs_format[i] == kOpFormat_DEFAULT) || (outputs_format[i] == origin_format)) {
      continue;
    }
    auto out_shape = common::AnfAlgo::GetOutputInferShape(node, i);
    auto dim = out_shape.size();
    if (dim < kMinDimNeedToTransform) {
      continue;
    }
    auto transpose_perm = TransposeAxis(dim, ChannelLast2ChannelFirst);
    // Find all nodes connected with node output, and change their inputs to transpose.
    auto used_node_list = GetRealNodeUsedListByOutputIdx(graph, node, i);
    for (size_t j = 0; j < used_node_list->size(); ++j) {
      auto used_node = used_node_list->at(j).first;
      auto used_node_index = used_node_list->at(j).second - 1;
      if (common::AnfAlgo::GetCNodeName(used_node) == prim::kPrimTupleGetItem->name()) {
        MS_LOG(DEBUG) << "The used node of [" << node->fullname_with_scope() << "] is tuple item.";
        // The tuple item need get next used nodes again.
        ProcessForTupleItem(graph, used_node, used_node_index, transpose_perm, outputs_format[i]);
        continue;
      }
      // node->used_node, if output format of node equals input format of used_node,
      // then no need to insert transpose between node and used_node.
      auto used_node_in_format = AnfUtils::IsRealCNodeKernel(used_node)
                                   ? AnfAlgo::GetInputFormat(used_node, IntToSize(used_node_index))
                                   : kOpFormat_DEFAULT;
      if (outputs_format[i] == used_node_in_format) {
        continue;
      }
      auto transpose_op = InsertTransposeOp(graph, node, used_node, used_node_index, transpose_perm);
      SetTransposeOpBuildInfo(outputs_format[i], kOpFormat_DEFAULT, transpose_op);
    }
  }
}
}  // namespace

const mindspore::HashSet<std::string> kChannelLastKernel = {prim::kBiasAdd};

bool InsertFormatTransformOpCPU::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());

  for (auto node : node_list) {
    if (!AnfUtils::IsRealCNodeKernel(node)) {
      continue;
    }

    auto iter = kChannelLastKernel.find(common::AnfAlgo::GetCNodeName(node));
    if (iter == kChannelLastKernel.end()) {
      continue;
    }
    auto origin_format = AnfAlgo::GetOriginDataFormat(node);
    if (origin_format == kOpFormat_DEFAULT) {
      origin_format = kOpFormat_ChannelFirst;
    }

    InsertTransformOpForInput(graph, node, origin_format);
    InsertTransformOpForOutput(graph, node, origin_format);
  }

  return true;
}
}  // namespace opt
}  // namespace mindspore
