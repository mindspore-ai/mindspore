/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "include/backend/distributed/ps/util.h"
#include <vector>
#include <memory>
#include "utils/hash_map.h"
#include "include/backend/distributed/ps/constants.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "utils/ms_utils.h"
#include "distributed/persistent/data.h"

namespace mindspore {
namespace ps {
namespace {
static mindspore::HashMap<std::string, int64_t> optimizer_to_ids = {
  {kApplyMomentum, 0},
  {kSparseAdam, 1},
  {kSparseLazyAdam, 2},
  {kSparseFtrl, 3},
};

static mindspore::HashMap<int64_t, std::string> id_to_optimizers = {
  {0, kApplyMomentum},
  {1, kSparseAdam},
  {2, kSparseLazyAdam},
  {3, kSparseFtrl},
};

static mindspore::HashMap<int64_t, std::string> id_to_optimizer_nodes = {
  {0, kApplyMomentumOp},
  {1, kSparseAdamOp},
  {2, kSparseLazyAdamOp},
  {3, kSparseFtrlOp},
};
}  // namespace

bool Util::IsRoleOfPServer() { return PSContext::instance()->is_server(); }

bool Util::IsRoleOfScheduler() { return PSContext::instance()->is_scheduler(); }

int64_t Util::optimizer_id(const std::string &name) {
  if (optimizer_to_ids.count(name) > 0) {
    return optimizer_to_ids[name];
  }
  return -1;
}

std::string Util::optimizer_name(int64_t id) {
  if (id_to_optimizers.count(id) > 0) {
    return id_to_optimizers[id];
  }
  return "";
}

std::string Util::optimizer_node_name(int64_t id) {
  if (id_to_optimizer_nodes.count(id) > 0) {
    return id_to_optimizer_nodes[id];
  }
  return "";
}

bool Util::is_optimizer(const std::string &name) { return optimizer_to_ids.count(name) > 0; }

int64_t Util::LocalShard(int64_t first_dim, int64_t rank_id, int64_t server_num) {
  std::map<int64_t, int64_t> shard_dims = AllRankLocalShard(first_dim, rank_id, server_num);
  if (shard_dims.count(rank_id) == 0) {
    MS_LOG(EXCEPTION) << "Invalid rank id " << rank_id;
  }
  return shard_dims[rank_id];
}

std::map<int64_t, int64_t> Util::AllRankLocalShard(int64_t first_dim, int64_t rank_id, int64_t server_num) {
  if (first_dim <= 0 || server_num <= 0 || rank_id < 0) {
    MS_LOG(EXCEPTION) << "Input values are invalid, first_dim: " << first_dim << ", server_num: " << server_num
                      << ", rank_id: " << rank_id;
  }
  if (rank_id >= server_num) {
    MS_LOG(EXCEPTION) << "The rank ID " << rank_id << " should be less than the number of servers " << server_num;
  }
  std::map<int64_t, int64_t> shard_dims;
  for (int64_t i = 0; i < server_num; i++) {
    shard_dims[i] = 0;
  }
  if (server_num != static_cast<int64_t>(shard_dims.size())) {
    MS_LOG(EXCEPTION) << "Inconsistent server num " << server_num << " shard dims counter size " << shard_dims.size();
  }
  int64_t server_index = -1;
  for (int64_t i = 0; i < first_dim; i++) {
    server_index = (server_index + 1) % server_num;
    shard_dims[server_index] = shard_dims[server_index] + 1;
  }
  if (shard_dims.count(rank_id) == 0) {
    MS_LOG(EXCEPTION) << "Invalid rank id " << rank_id << ", total server num " << server_num;
  }
  return shard_dims;
}

bool Util::FuseServerCommOps(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  DoFusion(func_graph, kPullWeightOpName, kFusedPullWeightOpName);
  DoFusion(func_graph, kPushWeightOpName, kFusedPushWeightOpName);
  return true;
}

WeightPtr Util::MakeWeightPtr(const std::shared_ptr<std::vector<float>> &data, bool enable_recovery,
                              const std::shared_ptr<std::vector<int>> &shape) {
  WeightPtr weight_ptr;
  if (!enable_recovery) {
    weight_ptr = std::make_shared<Weight>(data, shape);
  } else {
    weight_ptr = std::make_shared<PersistentWeight>(data, shape);
  }
  return weight_ptr;
}

std::string Util::GetPrimitiveName(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs of node " << cnode->fullname_with_scope() << " is empty.";
    return "";
  }
  auto fn = inputs[0];
  if (!IsValueNode<Primitive>(fn)) {
    return "";
  }

  auto node_prim = GetValueNode<PrimitivePtr>(fn);
  MS_EXCEPTION_IF_NULL(node_prim);
  return node_prim->name();
}

void Util::DoFusion(const FuncGraphPtr &func_graph, const std::string &cnode_name,
                    const std::string &fused_cnode_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());

  std::vector<AnfNodePtr> single_nodes;
  std::vector<std::string> weight_names;
  std::vector<int64_t> indices;
  for (const AnfNodePtr &node : node_list) {
    if (node != nullptr && node->isa<CNode>()) {
      if (GetPrimitiveName(node->cast<CNodePtr>()) == cnode_name) {
        single_nodes.push_back(node);

        auto weight_name_value_node =
          common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), kNodeInputWeightNameOffset)->cast<ValueNodePtr>();
        const std::string &weight_name = GetValue<std::string>(weight_name_value_node->value());
        weight_names.push_back(weight_name);

        auto weight_index_value_node =
          common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), kNodeInputWeightIndexOffset)->cast<ValueNodePtr>();
        int64_t weight_index = GetValue<int64_t>(weight_index_value_node->value());
        indices.push_back(weight_index);
      }
    }
  }

  auto prim = std::make_shared<Primitive>(fused_cnode_name);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> fused_node_inputs = {};
  fused_node_inputs.push_back(NewValueNode(prim));
  (void)std::for_each(single_nodes.begin(), single_nodes.end(), [&](const AnfNodePtr &node) {
    fused_node_inputs.push_back(common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), 0));
  });

  auto fused_cnode = func_graph->NewCNode(fused_node_inputs);
  MS_EXCEPTION_IF_NULL(fused_cnode);
  common::AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(weight_names), fused_cnode);
  common::AnfAlgo::SetNodeAttr(kAttrIndex, MakeValue(indices), fused_cnode);
  common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(kCPUDevice), fused_cnode);

  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  fused_cnode->set_kernel_info(kernel_info);
  auto kernel_build_info = GenerateKernelBuildInfo(single_nodes);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, fused_cnode.get());

  AbstractBasePtrList abstract_list;
  for (const auto &node : single_nodes) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    abstract_list.push_back(cnode->abstract());
  }
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  fused_cnode->set_abstract(abstract_tuple);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &node : single_nodes) {
    if (!manager->Replace(node, fused_cnode)) {
      MS_LOG(EXCEPTION) << "manager replace node failed";
    }
  }
  return;
}

kernel::KernelBuildInfoPtr Util::GenerateKernelBuildInfo(const std::vector<AnfNodePtr> &node_list) {
  std::vector<std::string> inputs_device_format;
  std::vector<std::string> outputs_device_format;
  std::vector<TypeId> inputs_device_type;
  std::vector<TypeId> outputs_device_type;
  std::vector<ShapeVector> outputs_shape;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  for (size_t idx = 0; idx < node_list.size(); ++idx) {
    auto cnode = utils::cast<CNodePtr>(node_list[idx]);
    MS_EXCEPTION_IF_NULL(cnode);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      (void)inputs_device_format.emplace_back(kOpFormat_DEFAULT);
      inputs_device_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_index));
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      (void)outputs_device_format.emplace_back(kOpFormat_DEFAULT);
      outputs_device_type.push_back(common::AnfAlgo::GetOutputInferDataType(cnode, output_index));
      outputs_shape.push_back(common::AnfAlgo::GetOutputInferShape(cnode, output_index));
    }
  }
  builder.SetInputsFormat(inputs_device_format);
  builder.SetOutputsFormat(outputs_device_format);
  builder.SetInputsDeviceType(inputs_device_type);
  builder.SetOutputsDeviceType(outputs_device_type);
  return builder.Build();
}
}  // namespace ps
}  // namespace mindspore
