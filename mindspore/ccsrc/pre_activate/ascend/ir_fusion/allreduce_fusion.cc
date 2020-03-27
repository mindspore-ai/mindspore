/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "pre_activate/ascend/ir_fusion/allreduce_fusion.h"

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "utils/utils.h"
#include "utils/graph_utils.h"
#include "operator/ops.h"
#include "device/kernel_info.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/kernel_build_info.h"
#include "parallel/context.h"

namespace mindspore {
namespace opt {
namespace {
kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(const AllReduceInfo_t &allreduce_node_info, size_t start_index,
                                                   size_t end_index) {
  if (end_index >= allreduce_node_info.allreduce_node.size()) {
    MS_LOG(EXCEPTION) << "end index out of vector size";
  }
  std::vector<std::string> inputs_device_format;
  std::vector<std::string> outputs_device_format;
  std::vector<TypeId> inputs_device_type;
  std::vector<TypeId> outputs_device_type;
  std::vector<std::vector<size_t>> outputs_shape;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  for (size_t idx = start_index; idx <= end_index; ++idx) {
    auto cnode = allreduce_node_info.allreduce_node[idx];
    MS_EXCEPTION_IF_NULL(cnode);
    for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(cnode); ++input_index) {
      inputs_device_format.push_back(AnfAlgo::GetInputFormat(cnode, input_index));
      inputs_device_type.push_back(AnfAlgo::GetInputDeviceDataType(cnode, input_index));
    }
    for (size_t output_index = 0; output_index < AnfAlgo::GetOutputTensorNum(cnode); ++output_index) {
      outputs_device_format.push_back(AnfAlgo::GetOutputFormat(cnode, output_index));
      outputs_device_type.push_back(AnfAlgo::GetOutputDeviceDataType(cnode, output_index));
      outputs_shape.push_back(AnfAlgo::GetOutputInferShape(cnode, output_index));
    }
    builder.SetFusionType(AnfAlgo::GetFusionType(cnode));
    builder.SetProcessor(AnfAlgo::GetProcessor(cnode));
    builder.SetKernelType(AnfAlgo::GetKernelType(cnode));
  }
  builder.SetInputsFormat(inputs_device_format);
  builder.SetOutputsFormat(outputs_device_format);
  builder.SetInputsDeviceType(inputs_device_type);
  builder.SetOutputsDeviceType(outputs_device_type);
  return builder.Build();
}
}  // namespace

bool AllReduceFusion::GetSplitSegments(const AllReduceInfo_t &allreduce_node_info, size_t *segment_num,
                                       std::vector<size_t> *segment_index) const {
  MS_EXCEPTION_IF_NULL(segment_num);
  MS_EXCEPTION_IF_NULL(segment_index);
  size_t allreduce_node_size = allreduce_node_info.allreduce_node.size();
  MS_LOG(INFO) << "graph all reduce node size " << allreduce_node_size;

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  const std::vector<uint32_t> split_indices = parallel_context->all_reduce_fusion_split_indices();

  size_t segments = 0;
  if (split_indices.size() != 0) {
    uint32_t last_index = 0;
    for (size_t i = 0; i < split_indices.size(); ++i) {
      uint32_t index = split_indices[i];
      if (index <= last_index || index >= allreduce_node_size) {
        MS_LOG(EXCEPTION) << "invalid allreduce split index " << i << " " << index;
      }
      segment_index->push_back(index);
      last_index = index;
      segments++;
    }
    if (last_index != allreduce_node_size - 1) {
      segment_index->push_back(allreduce_node_size - 1);
      segments++;
    }
  } else {
    segments = groups_;
    for (size_t i = 0; i < segments - 1; ++i) {
      segment_index->push_back((i + 1) * (allreduce_node_size / segments) - 1);
    }
    segment_index->push_back(allreduce_node_size - 1);
  }

  if (segments >= allreduce_node_size) {
    MS_LOG(INFO) << "fusion not changed: segment_num=" << segments << ", allreduce_node_size=" << allreduce_node_size;
    return false;
  }
  if (segment_index->at(segments - 1) != allreduce_node_size - 1) {
    MS_LOG(EXCEPTION) << "the last segment index is invalid.";
  }
  for (size_t i = 0; i < segments - 1; ++i) {
    if (segment_index->at(i) > segment_index->at(i + 1)) {
      MS_LOG(EXCEPTION) << "illegal split: segment_index[" << i << "]=" << segment_index->at(i) << ", segment_index[ "
                        << i + 1 << "]=" << segment_index->at(i + 1);
    }
  }
  *segment_num = segments;
  return true;
}

AnfNodePtr AllReduceFusion::CreateFusedAllReduce(const FuncGraphPtr &func_graph,
                                                 const AllReduceInfo_t &allreduce_node_info, size_t start_index,
                                                 size_t end_index) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto prim = std::make_shared<Primitive>(kAllReduceOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> fusion_inputs = {NewValueNode(prim)};
  // get all inputs of current segment
  if (end_index >= allreduce_node_info.allreduce_node.size()) {
    MS_LOG(EXCEPTION) << "end index out of vector size";
  }
  for (size_t idx = start_index; idx <= end_index; ++idx) {
    auto cnode = allreduce_node_info.allreduce_node[idx];
    MS_EXCEPTION_IF_NULL(cnode);
    fusion_inputs.insert(fusion_inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
  }
  AnfNodePtr fused_node = func_graph->NewCNode(fusion_inputs);
  MS_EXCEPTION_IF_NULL(fused_node);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  fused_node->set_kernel_info(kernel_info);
  AbstractBasePtrList abstract_list;
  for (size_t idx = start_index; idx <= end_index; ++idx) {
    auto cnode = allreduce_node_info.allreduce_node[idx];
    MS_EXCEPTION_IF_NULL(cnode);
    AnfAlgo::CopyNodeAttr("fusion", cnode, fused_node);
    AnfAlgo::CopyNodeAttr("op", cnode, fused_node);
    AnfAlgo::CopyNodeAttr("group", cnode, fused_node);
    abstract_list.push_back(cnode->abstract());
  }
  auto kernel_build_info = GenerateKernelBuildInfo(allreduce_node_info, start_index, end_index);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, fused_node.get());
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  fused_node->set_abstract(abstract_tuple);
  return fused_node;
}

bool AllReduceFusion::DoFusion(const FuncGraphPtr &func_graph, const AllReduceInfo_t &allreduce_node_info,
                               size_t segment_num, const std::vector<size_t> &segment_index) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  bool changed = false;
  size_t start_index = 0;
  for (size_t segment_idx = 0; segment_idx < segment_num; ++segment_idx) {
    size_t end_index = segment_index.at(segment_idx);
    if (end_index - start_index < 1) {
      start_index = end_index + 1;
      continue;
    }
    AnfNodePtr new_allreduce = CreateFusedAllReduce(func_graph, allreduce_node_info, start_index, end_index);
    // replace old allreduce with new allreduce
    for (auto idx = start_index; idx <= end_index; ++idx) {
      std::vector<AnfNodePtr> tuple_getitem_input;
      tuple_getitem_input.push_back(NewValueNode(prim::kPrimTupleGetItem));
      tuple_getitem_input.push_back(new_allreduce);
      auto index = NewValueNode(SizeToInt(idx - start_index));
      MS_EXCEPTION_IF_NULL(index);
      auto imm = std::make_shared<Int32Imm>(idx - start_index);
      MS_EXCEPTION_IF_NULL(imm);
      auto abstract_scalar = std::make_shared<abstract::AbstractScalar>();
      MS_EXCEPTION_IF_NULL(abstract_scalar);
      index->set_abstract(abstract_scalar);
      tuple_getitem_input.push_back(index);
      AnfNodePtr tuple_getitem = func_graph->NewCNode(tuple_getitem_input);
      MS_EXCEPTION_IF_NULL(tuple_getitem);
      auto allreduce_node_item = allreduce_node_info.allreduce_node.at(idx);
      MS_EXCEPTION_IF_NULL(allreduce_node_item);
      tuple_getitem->set_abstract(allreduce_node_item->abstract());
      if (!manager->Replace(allreduce_node_item, tuple_getitem)) {
        MS_LOG(EXCEPTION) << "manager replace node failed";
      }
    }
    start_index = end_index + 1;
    changed = true;
  }
  return changed;
}

bool AllReduceFusion::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const float input_grad_size_num = 0.0;
  const float input_grad_time_num = 0.0;
  // divide candidate fusion groups with same (group,op,fusion) attrs, fusion==0 means not fusion
  std::unordered_map<std::string, AllReduceInfo_t> candidate_groups;
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>() && AnfAlgo::GetCNodeName(node) == kAllReduceOpName) {
      auto primitive = AnfAlgo::GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(primitive);
      int fusion = GetValue<int>(primitive->GetAttr("fusion"));
      if (fusion == 0) {
        continue;
      }
      std::string group = GetValue<std::string>(primitive->GetAttr("group"));
      std::string op = GetValue<std::string>(primitive->GetAttr("op"));
      std::string key = group + op + std::to_string(fusion);
      if (candidate_groups.find(key) == candidate_groups.end()) {
        AllReduceInfo_t allreduce_node_info;
        candidate_groups[key] = allreduce_node_info;
      }
      candidate_groups[key].allreduce_node.push_back(node->cast<CNodePtr>());
      candidate_groups[key].input_grad_size.push_back(input_grad_size_num);
      candidate_groups[key].input_grad_time.push_back(input_grad_time_num);
    }
  }
  // split candidate group to segments according to _group class member
  bool changed = false;
  for (auto &it : candidate_groups) {
    if (it.second.allreduce_node.size() <= 1) {
      continue;
    }
    size_t segment_num = 0;
    std::vector<size_t> segment_index;
    if (GetSplitSegments(it.second, &segment_num, &segment_index)) {
      if (DoFusion(func_graph, it.second, segment_num, segment_index)) {
        changed = true;
      }
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
