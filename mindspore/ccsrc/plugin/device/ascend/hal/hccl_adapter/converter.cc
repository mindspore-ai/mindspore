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
#include "plugin/device/ascend/hal/hccl_adapter/converter.h"
#include <map>
#include <algorithm>
#include <tuple>
#define google ascend_private
#include "register/ops_kernel_builder_registry.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#undef google
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/log_adapter.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/comm_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/all_to_all_v_calc_param.h"

namespace mindspore::hccl {
namespace {
static constexpr char kGeOpNameHcclSend[] = "HcomSend";
static constexpr char kGeOpNameHcclReceive[] = "HcomReceive";
static constexpr char kGeOpNameHcclAllRudece[] = "HcomAllReduce";
static constexpr char kGeOpNameHcclAllGather[] = "HcomAllGather";
static constexpr char kGeOpNameHcclBroadcast[] = "HcomBroadcast";
static constexpr char kGeOpNameHcclReduceScatter[] = "HcomReduceScatter";
static constexpr char kGeOpNameHcclAllToAllV[] = "HcomAllToAllV";
static constexpr char kGeNodeAttrUsedStreamNum[] = "used_stream_num";
static constexpr char kGeNodeAttrSendCounts[] = "send_counts";
static constexpr char kGeNodeAttrSendDispls[] = "send_displacements";
static constexpr char kGeNodeAttrRecvCounts[] = "recv_counts";
static constexpr char kGeNodeAttrRecvDispls[] = "recv_displacements";
static constexpr char kStubDataStructureName[] = "any_name_can_work";

static ge::DataType ConvertHcclDTypeToGeDType(HcclDataType datatype) {
  static map<HcclDataType, ge::DataType> kHcomDataTypeMap = {
    {HCCL_DATA_TYPE_FP32, ge::DT_FLOAT},
    {HCCL_DATA_TYPE_FP16, ge::DT_FLOAT16},
    {HCCL_DATA_TYPE_INT8, ge::DT_INT8},
    {HCCL_DATA_TYPE_INT32, ge::DT_INT32},
  };

  decltype(kHcomDataTypeMap)::const_iterator iter = kHcomDataTypeMap.find(datatype);
  if (iter == kHcomDataTypeMap.end()) {
    MS_LOG(EXCEPTION) << "Unknown hccl data type " << datatype;
  }

  return iter->second;
}

template <class T>
struct IsString {
  // cppcheck-suppress unusedStructMember
  static constexpr bool value = false;
};

template <>
struct IsString<std::string> {
  // cppcheck-suppress unusedStructMember
  static constexpr bool value = true;
};

template <class T>
struct IsVector {
  // cppcheck-suppress unusedStructMember
  static constexpr bool value = false;
};

template <>
struct IsVector<std::vector<int64_t>> {
  // cppcheck-suppress unusedStructMember
  static constexpr bool value = true;
};
}  // namespace

template <class T>
static T ConvertAttr(const CNodePtr &cnode, const ge::OpDescPtr &ge_op, const std::string &anf_attr_name,
                     const std::string &ge_attr_name) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(ge_op);
  if (!common::AnfAlgo::HasNodeAttr(anf_attr_name, cnode)) {
    MS_LOG(INFO) << "Node " << cnode->DebugString() << " has no attr " << anf_attr_name << ", skip.";
    return T();
  }

  bool ret;
  auto attr = common::AnfAlgo::GetNodeAttr<T>(cnode, anf_attr_name);
  if constexpr (IsString<T>::value) {
    ret = ge::AttrUtils::SetStr(*ge_op, ge_attr_name, attr);
  } else if constexpr (IsVector<T>::value) {
    ret = ge::AttrUtils::SetListInt(*ge_op, ge_attr_name, attr);
  } else {
    ret = ge::AttrUtils::SetInt(*ge_op, ge_attr_name, attr);
  }

  if (!ret) {
    MS_LOG(EXCEPTION) << "Set attr " << ge_attr_name << " for ge node of " << cnode->DebugString() << " failed.";
  }
  MS_LOG(INFO) << "Convert success, attr " << ge_attr_name << " is " << attr;
  return attr;
}

static void SetGeNodeInt64VecAttr(const ge::OpDescPtr &ge_op, const std::string &ge_attr_name,
                                  const std::vector<int64_t> &attr_value) {
  MS_EXCEPTION_IF_NULL(ge_op);
  for (size_t i = 0; i < attr_value.size(); ++i) {
    MS_LOG(INFO) << ge_attr_name << " " << i << " = " << attr_value[i];
  }
  auto ret = ge::AttrUtils::SetListInt(*ge_op, ge_attr_name, attr_value);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Set attr " << ge_attr_name << " for ge node failed.";
  }
}

static void SetAllToAllvAttr(const CNodePtr &cnode, const ge::OpDescPtr &ge_op, const std::string &group) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(ge_op);
  if (!IsPrimitiveCNode(cnode, prim::kPrimAllToAllv)) {
    return;
  }
  uint32_t rank_size = 0;
  if (!CommManager::GetInstance().GetRankSize(group, &rank_size)) {
    MS_LOG(EXCEPTION) << "Get hccl rank size for group " << group << " failed.";
  }
  mindspore::hccl::AllToAllvCalcParam calc(cnode, rank_size);
  calc.CalcOpParam();
  SetGeNodeInt64VecAttr(ge_op, kGeNodeAttrSendCounts, calc.GetSendCounts());
  SetGeNodeInt64VecAttr(ge_op, kGeNodeAttrSendDispls, calc.GetSendDispls());
  SetGeNodeInt64VecAttr(ge_op, kGeNodeAttrRecvCounts, calc.GetRecvCounts());
  SetGeNodeInt64VecAttr(ge_op, kGeNodeAttrRecvDispls, calc.GetRecvDispls());
}

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
  } else if (IsPrimitiveCNode(cnode, prim::kPrimSend)) {
    return kGeOpNameHcclSend;
  } else if (IsPrimitiveCNode(cnode, prim::kPrimReceive)) {
    return kGeOpNameHcclReceive;
  } else if (IsPrimitiveCNode(cnode, prim::kPrimAllToAllv)) {
    return kGeOpNameHcclAllToAllV;
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
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t i = 0; i < input_num; ++i) {
    auto ge_shape = AnfAlgo::GetInputDeviceShape(cnode, i);
    (void)op_desc->AddInputDesc(
      ge::GeTensorDesc(ge::GeShape(ge_shape), ge::Format::FORMAT_NCHW,
                       transform::ConvertDataType(AnfAlgo::GetInputDeviceDataType(cnode, i))));
  }
  for (size_t i = 0; i < output_num; ++i) {
    auto ge_shape = AnfAlgo::GetOutputDeviceShape(cnode, i);
    (void)op_desc->AddOutputDesc(
      ge::GeTensorDesc(ge::GeShape(ge_shape), ge::Format::FORMAT_NCHW,
                       transform::ConvertDataType(AnfAlgo::GetOutputDeviceDataType(cnode, i))));
  }

  // set node data type
  bool ret = ge::AttrUtils::SetDataType(*op_desc, ge::HCOM_ATTR_DATA_TYPE, ConvertHcclDTypeToGeDType(datatype));
  if (!ret) {
    MS_LOG(EXCEPTION) << "Set attr " << ge::HCOM_ATTR_DATA_TYPE << " for ge node of " << cnode->DebugString()
                      << " failed.";
  }

  // set node attr
  (void)ConvertAttr<int64_t>(cnode, op_desc, kAttrRankSize, ge::HCOM_ATTR_RANK_SIZE);
  auto group = ConvertAttr<std::string>(cnode, op_desc, kAttrGroup, ge::HCOM_ATTR_GROUP);
  (void)ConvertAttr<int64_t>(cnode, op_desc, kAttrSrcRank, ge::HCOM_ATTR_SRC_RANK);
  (void)ConvertAttr<int64_t>(cnode, op_desc, kAttrDestRank, ge::HCOM_ATTR_DEST_RANK);
  (void)ConvertAttr<int64_t>(cnode, op_desc, kAttrSrTag, ge::HCOM_ATTR_SR_TAG);
  (void)ConvertAttr<int64_t>(cnode, op_desc, kAttrComm, "comm");
  (void)ConvertAttr<std::vector<int64_t>>(cnode, op_desc, kAttrShape, ge::HCOM_ATTR_SHAPE);
  SetAllToAllvAttr(cnode, op_desc, group);

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
