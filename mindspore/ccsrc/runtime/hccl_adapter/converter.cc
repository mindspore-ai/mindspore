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
#include "runtime/hccl_adapter/converter.h"
#include <map>
#include <algorithm>
#include <tuple>
#define google ascend_private
#include "register/ops_kernel_builder_registry.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#undef google
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "mindspore/core/base/core_ops.h"
#include "transform/graph_ir/util.h"

static constexpr char kGeOpNameHcclAllRudece[] = "HcomAllReduce";
static constexpr char kGeOpNameHcclAllGather[] = "HcomAllGather";
static constexpr char kGeOpNameHcclBroadcast[] = "HcomBroadcast";
static constexpr char kGeOpNameHcclReduceScatter[] = "HcomReduceScatter";
static constexpr char kGeNodeAttrUsedStreamNum[] = "used_stream_num";
static constexpr char kStubDataStructureName[] = "any_name_can_work";

static ge::DataType ConvertHcclDTypeToGeDType(HcclDataType datatype) {
  static map<HcclDataType, ge::DataType> kHcomDataTypeMap = {
    {HCCL_DATA_TYPE_FP32, ge::DT_FLOAT},
    {HCCL_DATA_TYPE_FP16, ge::DT_FLOAT16},
    {HCCL_DATA_TYPE_INT8, ge::DT_INT8},
    {HCCL_DATA_TYPE_INT32, ge::DT_INT32},
  };

  auto iter = kHcomDataTypeMap.find(datatype);
  if (iter == kHcomDataTypeMap.end()) {
    MS_LOG(EXCEPTION) << "Unknown hccl data type " << datatype;
  }

  return iter->second;
}

namespace mindspore::hccl {
std::string GetGeNodeName(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsPrimitiveCNode(cnode, prim::kPrimAllReduce)) {
    return kGeOpNameHcclAllRudece;
  } else if (IsPrimitiveCNode(cnode, prim::kPrimAllGather)) {
    return kGeOpNameHcclAllGather;
  } else if (IsPrimitiveCNode(cnode, prim::kPrimBroadcast)) {
    return kGeOpNameHcclBroadcast;
  } else if (IsPrimitiveCNode(cnode, prim::kPrimReduceScatter)) {
    return kGeOpNameHcclReduceScatter;
  }

  MS_LOG(EXCEPTION) << "Unknown hccl node type " << cnode->DebugString();
}

std::tuple<ge::NodePtr, ge::ComputeGraphPtr> GenerateStubGeNode(const AnfNodePtr &anf_node, HcclDataType datatype) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::string ge_node_name = GetGeNodeName(cnode);

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>(kStubDataStructureName, ge_node_name);
  MS_EXCEPTION_IF_NULL(op_desc);
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto &input = cnode->input(i);
    std::vector<int64_t> ge_shape;
    auto ms_shape = AnfAlgo::GetOutputInferShape(input, 0);
    std::transform(ms_shape.begin(), ms_shape.end(), std::back_inserter(ge_shape),
                   [](size_t in) { return static_cast<int64_t>(in); });
    op_desc->AddInputDesc(
      ge::GeTensorDesc(ge::GeShape(ge_shape), ge::Format::FORMAT_NCHW,
                       transform::TransformUtil::ConvertDataType(AnfAlgo::GetOutputInferDataType(input, 0))));
  }

  // set node data type
  bool ret = ge::AttrUtils::SetDataType(*op_desc, ge::HCOM_ATTR_DATA_TYPE, ConvertHcclDTypeToGeDType(datatype));
  if (!ret) {
    MS_LOG(EXCEPTION) << "Set attr " << ge::HCOM_ATTR_DATA_TYPE << " for ge node of " << cnode->DebugString()
                      << " failed.";
  }

  // set rank size
  if (AnfAlgo::HasNodeAttr(kAttrRankSize, cnode)) {
    auto rank_size = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrRankSize);
    ret = ge::AttrUtils::SetInt(*op_desc, ge::HCOM_ATTR_RANK_SIZE, rank_size);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Set attr " << ge::HCOM_ATTR_RANK_SIZE << " for ge node of " << cnode->DebugString()
                        << " failed.";
    }
  }

  ge::ComputeGraphPtr ge_graph = std::make_shared<ge::ComputeGraph>(kStubDataStructureName);
  MS_EXCEPTION_IF_NULL(ge_graph);
  auto ge_node = ge_graph->AddNode(op_desc);
  return {ge_node, ge_graph};
}

HcclTaskInfo ParseDomiTask(const ge::OpDescPtr &op, const domi::TaskDef &task_def) {
  MS_EXCEPTION_IF_NULL(op);
  // workspace size
  auto workspace_sizes = op->GetWorkspaceBytes();
  if (workspace_sizes.size() != 1) {
    MS_LOG(EXCEPTION) << "Unexpected workspace size " << workspace_sizes.size();
  }
  int64_t workspace_size = workspace_sizes[0];
  // stream num
  int64_t stream_num;
  bool ret = ge::AttrUtils::GetInt(*op, kGeNodeAttrUsedStreamNum, stream_num);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Get attr " << kGeNodeAttrUsedStreamNum << " for ge node " << op->GetType() << " failed.";
  }

  return {task_def.private_def(), workspace_size, stream_num};
}
}  // namespace mindspore::hccl
