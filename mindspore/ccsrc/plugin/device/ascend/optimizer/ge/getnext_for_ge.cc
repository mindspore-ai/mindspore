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

#include "plugin/device/ascend/optimizer/ge/getnext_for_ge.h"
#include <vector>
#include <memory>
#include <string>
#include "ops/other_op_name.h"
#include "ops/structure_ops.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr char kIndexAttrName[] = "index";
constexpr char kQueueNameAttrName[] = "queue_name";
constexpr char kSharedNameAttrName[] = "shared_name";
constexpr char kChannelNameAttrName[] = "channel_name";
constexpr char kOutputTypesAttrName[] = "output_types";
constexpr char kTypesAttrName[] = "types";
constexpr char kOutputShapesAttrName[] = "output_shapes";
constexpr char kShapesAttrName[] = "shapes";
constexpr char kExecuteModeAttrName[] = "_dynamic_graph_execute_mode";
constexpr char kInputsShapeRangeAttrName[] = "_getnext_inputs_shape_range";
}  // namespace

const BaseRef GetNextForGE::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimGetNext, Xs});
}

const AnfNodePtr ProcessGetNextForHeterogenous(const FuncGraphPtr &graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(cnode);

  // create QueueData node
  auto queue_data_node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kQueueDataOpName))});
  MS_EXCEPTION_IF_NULL(queue_data_node);
  int64_t index = 0;
  common::AnfAlgo::SetNodeAttr(kIndexAttrName, MakeValue(index), queue_data_node);
  common::AnfAlgo::CopyNodeAttr(kSharedNameAttrName, kQueueNameAttrName, cnode, queue_data_node);
  common::AnfAlgo::CopyNodeAttr(kTypesAttrName, kOutputTypesAttrName, cnode, queue_data_node);
  common::AnfAlgo::CopyNodeAttr(kShapesAttrName, kOutputShapesAttrName, cnode, queue_data_node);
  std::vector<int64_t> queue_data_shape = {1};
  auto queue_data_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, queue_data_shape);
  queue_data_node->set_abstract(queue_data_abstract);

  // Create GetNextFromQueue node
  auto getnext_from_queue_node =
    graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kGetNextFromQueueOpName)), queue_data_node});
  MS_EXCEPTION_IF_NULL(getnext_from_queue_node);
  common::AnfAlgo::CopyNodeAttr(kTypesAttrName, kOutputTypesAttrName, cnode, getnext_from_queue_node);
  common::AnfAlgo::CopyNodeAttr(kShapesAttrName, kOutputShapesAttrName, cnode, getnext_from_queue_node);
  getnext_from_queue_node->set_abstract(cnode->abstract());

  return getnext_from_queue_node;
}

bool IsDynamicGetNext(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::HasNodeAttr(kShapesAttrName, node)) {
    MS_LOG(INTERNAL_EXCEPTION) << "The GetNext node do not have attr " << kShapesAttrName
                               << ", node: " << node->fullname_with_scope();
  }
  auto shapes = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(node, kShapesAttrName);
  for (auto shape : shapes) {
    if (std::any_of(shape.cbegin(), shape.cend(), [](const auto e) { return e == -1; })) {
      return true;
    }
  }
  return false;
}

std::string GetShapesRange(const std::vector<std::vector<int64_t>> &shapes) {
  std::stringstream buffer;
  for (auto shape_it = shapes.begin(); shape_it != shapes.end(); ++shape_it) {
    if (shape_it != shapes.begin()) {
      buffer << ",";
    }
    buffer << "[";
    const auto &dims = *shape_it;
    for (auto dim_it = dims.begin(); dim_it != dims.end(); ++dim_it) {
      if (dim_it != dims.begin()) {
        buffer << ",";
      }
      buffer << *dim_it;
    }
    buffer << "]";
  }
  return buffer.str();
}

const AnfNodePtr ProcessGetNextForDynamicShape(const FuncGraphPtr &graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(cnode);

  auto dynamic_getnextv2_node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kDynamicGetNextV2OpName))});
  MS_EXCEPTION_IF_NULL(dynamic_getnextv2_node);
  common::AnfAlgo::CopyNodeAttr(kTypesAttrName, kOutputTypesAttrName, cnode, dynamic_getnextv2_node);
  common::AnfAlgo::CopyNodeAttr(kShapesAttrName, kOutputShapesAttrName, cnode, dynamic_getnextv2_node);
  common::AnfAlgo::CopyNodeAttr(kSharedNameAttrName, kChannelNameAttrName, cnode, dynamic_getnextv2_node);
  common::AnfAlgo::SetNodeAttr(kExecuteModeAttrName, MakeValue("dynamic_execute"), dynamic_getnextv2_node);
  auto input_shapes = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(cnode, kShapesAttrName);
  common::AnfAlgo::SetNodeAttr(kInputsShapeRangeAttrName, MakeValue(GetShapesRange(input_shapes)),
                               dynamic_getnextv2_node);
  dynamic_getnextv2_node->set_abstract(cnode->abstract());
  return dynamic_getnextv2_node;
}

// Set the attr dtype and convert it to ge_dtype
const AnfNodePtr GetNextForGE::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  if (kernel_graph != nullptr && kernel_graph->is_from_single_op()) {
    MS_LOG(INFO) << "Run GetNext by ACL and skip this process";
    return nullptr;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    return ProcessGetNextForHeterogenous(graph, cnode);
  } else if (IsDynamicGetNext(cnode)) {
    return ProcessGetNextForDynamicShape(graph, cnode);
  } else {
    return nullptr;
  }
}
}  // namespace opt
}  // namespace mindspore
