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

#include "backend/ge_backend/pass/all_to_all_v_for_ge.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <tuple>
#include <algorithm>
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "primitive/other_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAttrIsInsertedByGE = "is_inserted_by_ge";

std::vector<int64_t> GetNumeList(const CNodePtr &origin_node, size_t index) {
  auto input = common::AnfAlgo::GetInputNode(origin_node, index);

  MS_EXCEPTION_IF_NULL(input);
  auto input_abstract = input->abstract();
  MS_EXCEPTION_IF_NULL(input_abstract);
  auto input_value = input_abstract->GetValue();
  MS_EXCEPTION_IF_NULL(input_value);
  auto array = GetArrayValue<int64_t>(input_value);
  if (!array.has_value()) {
    MS_LOG(INFO) << "Array has no value.";
    return {};
  }
  auto array_value = array.value();
  if (array_value.HasUnknownValue()) {
    MS_LOG(INFO) << "Array_value has unknown value.";
    return {};
  }
  return array_value.ToVector();
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
GetAlltoAllVForGEInput(const CNodePtr &origin_node) {
  auto send_numel_list = GetNumeList(origin_node, kIndex1);
  auto recv_numel_list = GetNumeList(origin_node, kIndex2);
  auto send_offset_list = common::AnfAlgo::HasNodeAttr(kAttrSendOffsetList, origin_node)
                            ? common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(origin_node, kAttrSendOffsetList)
                            : std::vector<int64_t>{};
  auto recv_offset_list = common::AnfAlgo::HasNodeAttr(kAttrRecvOffsetList, origin_node)
                            ? common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(origin_node, kAttrRecvOffsetList)
                            : std::vector<int64_t>{};
  if (send_offset_list.empty()) {
    int64_t send_offset = 0;
    for (size_t i = 0; i < send_numel_list.size(); i++) {
      send_offset_list.push_back(send_offset);
      send_offset += send_numel_list[i];
    }
  }
  if (recv_offset_list.empty()) {
    int64_t recv_offset = 0;
    for (size_t i = 0; i < recv_numel_list.size(); i++) {
      recv_offset_list.push_back(recv_offset);
      recv_offset += recv_numel_list[i];
    }
  }
  return {send_numel_list, send_offset_list, recv_numel_list, recv_offset_list};
}
}  // namespace

CNodePtr AlltoAllVForGE::CreateAlltoAllVForGENode(const FuncGraphPtr &graph, const CNodePtr &origin_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  auto [send_numel_list, send_offset_list, recv_numel_list, recv_offset_list] = GetAlltoAllVForGEInput(origin_node);

  AnfNodePtrList atav_inputs = {NewValueNode(std::make_shared<Primitive>(kAlltoAllVGEOpName)),
                                common::AnfAlgo::GetInputNode(origin_node, kIndex0),
                                CreateShapeValueNode(graph, send_numel_list),
                                CreateShapeValueNode(graph, send_offset_list),
                                CreateShapeValueNode(graph, recv_numel_list),
                                CreateShapeValueNode(graph, recv_offset_list)};
  auto atav_node = NewCNode(atav_inputs, graph);
  MS_EXCEPTION_IF_NULL(atav_node);
  atav_node->set_scope(origin_node->scope());
  common::AnfAlgo::CopyNodeAttrs(origin_node, atav_node);
  auto data_type = common::AnfAlgo::GetOutputInferDataType(origin_node, kIndex0);
  // ge do not support None data type
  if (data_type == TypeId::kMetaTypeNone) {
    data_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, kIndex0);
  }
  auto shape = common::AnfAlgo::GetOutputInferShape(origin_node, kIndex0);
  common::AnfAlgo::SetOutputInferTypeAndShape({data_type}, {shape}, atav_node.get());
  if (shape.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrIsInsertedByGE, MakeValue(true), atav_node);
  }
  MS_LOG(INFO) << "Create AlltoAllVGE node: " << atav_node->fullname_with_scope()
               << " success, send counts: " << send_numel_list << ", send displacements: " << send_offset_list
               << ", recv counts: " << recv_numel_list << ", recv displacements: " << recv_offset_list;
  return atav_node;
}

std::vector<std::string> AlltoAllVForGE::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimAlltoAllV->name());
  return ret;
}

const BaseRef AlltoAllVForGE::DefinePattern() const {
  return VectorRef({prim::kPrimAlltoAllV, std::make_shared<SeqVar>()});
}

const AnfNodePtr AlltoAllVForGE::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto all_to_all_v = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  auto new_atav_node = CreateAlltoAllVForGENode(graph, all_to_all_v);
  return new_atav_node;
}
}  // namespace opt
}  // namespace mindspore
