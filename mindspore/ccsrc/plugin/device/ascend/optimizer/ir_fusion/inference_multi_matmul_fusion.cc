/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/multi_matmuls_fusion.h"
#include <vector>
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "include/api/data_type.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
tensor::TensorPtr GetParamFromLoad(const CNodePtr &load, const bool unused) {
  if (IsPrimitiveCNode(load, prim::kPrimLoad)) {
    auto anf_node = common::AnfAlgo::GetInputNode(load, kIndex0);
    MS_EXCEPTION_IF_NULL(anf_node);
    if (anf_node->isa<Parameter>()) {
      auto para = anf_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(para);
      if (para->has_default()) {
        auto value = para->default_param();
        MS_EXCEPTION_IF_NULL(value);
        auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
        MS_EXCEPTION_IF_NULL(tensor);
        if (unused) {
          auto param_info = para->param_info();
          param_info->set_ignore_device_addr(true);
        }
        return tensor;
      }
    }
  }
  return nullptr;
}

bool CheckFusionValid(const CNodePtr &matmul, int64_t *k) {
  auto inputs = matmul->inputs();
  auto trans_a_node = GetValueNode(inputs[inputs.size() - 2]);
  auto trans_b_node = GetValueNode(inputs[inputs.size() - 1]);
  MS_EXCEPTION_IF_NULL(trans_a_node);
  MS_EXCEPTION_IF_NULL(trans_b_node);
  bool trans_a = GetValue<bool>(trans_a_node);
  bool trans_b = GetValue<bool>(trans_b_node);
  if (trans_a != false) return false;
  if (trans_b != true) return false;
  auto weight_node = inputs[kIndex2]->cast<CNodePtr>();
  auto w_param = GetParamFromLoad(weight_node, false);
  if (!w_param) return false;
  std::vector<int64_t> shape = w_param->shape();
  auto w_type_id = static_cast<TypeId>(w_param->data_type_c());
  if (w_type_id != TypeId::kNumberTypeBFloat16 && w_type_id != TypeId::kNumberTypeFloat16) {
    return false;
  }
  if (shape.size() != 2) return false;
  if (*k == -1) {
    *k = shape[1];
  } else if (*k != shape[1]) {
    return false;
  }
  return true;
}

template <typename T>
void ConcatWeightsToNewTensor(void *data_ptr, const std::vector<void *> &data_c_list, int64_t k_len,
                              const std::vector<int64_t> &n_len_list, bool need_rank_offset) {
  const auto data_size = sizeof(T);
  int64_t offset = 0;
  auto global_rank_id = distributed::collective::CollectiveManager::instance()->global_rank_id();
  for (int idx = 0; idx < static_cast<int>(data_c_list.size()); idx++) {
    auto count = k_len * n_len_list[idx];
    auto rank_offset = need_rank_offset ? global_rank_id * count : 0;
    std::memcpy(reinterpret_cast<T *>(data_ptr) + offset, reinterpret_cast<T *>(data_c_list[idx]) + rank_offset,
                count * data_size);
    offset += count;
  }
}

std::shared_ptr<ValueNode> CreateWeightTensor(TypeId type_id, const std::vector<int64_t> &weight_shape,
                                              const std::vector<void *> &data_c_list,
                                              const std::vector<int64_t> &n_len_list, int64_t k_len,
                                              const std::shared_ptr<Type> &w_dtype, bool need_rank_offset) {
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(type_id, weight_shape);
  auto data_ptr = assist_tensor->data_c();
  if (type_id == TypeId::kNumberTypeBFloat16) {
    ConcatWeightsToNewTensor<bfloat16>(data_ptr, data_c_list, k_len, n_len_list, need_rank_offset);
  } else if (type_id == TypeId::kNumberTypeFloat16) {
    ConcatWeightsToNewTensor<float16>(data_ptr, data_c_list, k_len, n_len_list, need_rank_offset);
  }
  TensorTypePtr tensor_type = std::make_shared<TensorType>(w_dtype);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  assist_tensor->set_device_info(device_info);
  MS_EXCEPTION_IF_NULL(assist_tensor);
  auto assist_const = std::make_shared<ValueNode>(assist_tensor);
  auto assist_abstract = assist_tensor->ToAbstract();
  assist_const->set_abstract(assist_abstract);
  auto assist_kernel_info = std::make_shared<device::KernelInfo>();
  assist_const->set_kernel_info(assist_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  builder.SetOutputsDeviceType({common::AnfAlgo::GetOutputInferDataType(assist_const, 0)});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), assist_const.get());
  return assist_const;
}
}  // namespace

bool InferenceMultiMatmulFusion::Run(const FuncGraphPtr &graph) {
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool changed = false;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return false;
  }

  constexpr auto kInferenceMultiMatmulOpName = "InferenceMultiMatmul";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_opt =
    (std::find(enable_op_list.begin(), enable_op_list.end(), kInferenceMultiMatmulOpName) != enable_op_list.end());
  if (!enable_opt) {
    return false;
  }

  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  const auto &node_users_map = mng->node_users();
  auto node_list = TopoSort(graph->output());
  constexpr size_t kMatMulFfnNum = 2;
  constexpr size_t kMatMulQkvNum = 3;

  int64_t k_len = -1;
  for (const auto &node : node_list) {
    AnfNodePtrList user_matmuls;
    if (node_users_map.find(node) == node_users_map.end()) continue;
    bool can_process = true;
    for (const auto &user_pair : node_users_map.at(node)) {
      // the node is MatMul's first input.
      if (IsPrimitiveCNode(user_pair.first, prim::kPrimMatMul) && user_pair.second == 1) {
        auto curr_matmul = user_pair.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(curr_matmul);
        can_process &= CheckFusionValid(curr_matmul, &k_len);
        user_matmuls.push_back(user_pair.first);
      }
    }
    if (user_matmuls.size() <= 1 || !can_process) {
      continue;
    }
    AnfNodePtrList getitems;
    if (user_matmuls.size() == kMatMulFfnNum) {
      Process("MatmulFfn", node, user_matmuls, &getitems);
    } else if (user_matmuls.size() == kMatMulQkvNum) {
      Process("MatmulQkv", node, user_matmuls, &getitems);
    }
    if (!getitems.empty()) {
      for (size_t i = 0; i < getitems.size(); i++) {
        (void)mng->Replace(user_matmuls[i], getitems[i]);
      }
      changed = true;
    }
  }
  return changed;
}

void InferenceMultiMatmulFusion::Process(const std::string &name, const AnfNodePtr &node, const AnfNodePtrList &users,
                                         AnfNodePtrList *getitems) const {
  auto kernel_graph = node->func_graph()->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  AnfNodePtrList fused_inputs = {NewValueNode(std::make_shared<Primitive>(name)), node};
  abstract::AbstractBasePtrList new_abs;

  TypeId w_type_id = kTypeUnknown;
  std::vector<void *> data_c_list;
  int64_t k_len = 0;
  int64_t n_len = 0;
  std::vector<int64_t> n_len_list;
  bool need_rank_offset = false;

  for (auto &user : users) {
    auto matmul = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(matmul);
    auto weight_node = matmul->inputs()[2]->cast<CNodePtr>();

    auto w_param = GetParamFromLoad(weight_node, true);
    MS_EXCEPTION_IF_NULL(w_param);
    data_c_list.push_back(reinterpret_cast<void *>(w_param->data_c()));
    auto origin_shape = w_param->shape();
    auto shape = common::AnfAlgo::GetOutputInferShape(weight_node, kIndex0);
    if (origin_shape[0] != shape[0]) {
      need_rank_offset = true;
    }
    k_len = shape[1];
    n_len += shape[0];
    n_len_list.push_back(shape[0]);
    w_type_id = static_cast<TypeId>(w_param->data_type_c());
    new_abs.push_back(user->abstract());
  }

  // Create a new weight tensor concat all of weights
  TypePtr w_dtype = (w_type_id == TypeId::kNumberTypeBFloat16) ? kBFloat16 : kFloat16;
  std::vector<int64_t> new_weight_shape = {n_len, k_len};
  std::shared_ptr<ValueNode> weight_value_node =
    CreateWeightTensor(w_type_id, new_weight_shape, data_c_list, n_len_list, k_len, w_dtype, need_rank_offset);
  fused_inputs.push_back(weight_value_node);
  kernel_graph->AddValueNodeToGraph(weight_value_node);

  auto fused_matmul = node->func_graph()->NewCNode(fused_inputs);
  common::AnfAlgo::SetNodeAttr("n_lens", MakeValue(n_len_list), fused_matmul);
  const bool is_fixed_weight = true;
  common::AnfAlgo::SetNodeAttr("is_fixed_weight", MakeValue<bool>(is_fixed_weight), fused_matmul);

  fused_matmul->set_abstract(std::make_shared<abstract::AbstractTuple>(new_abs));
  for (size_t i = 0; i < users.size(); i++) {
    auto idx_val = MakeValue(SizeToLong(i));
    auto idx = NewValueNode(idx_val);
    idx->set_abstract(idx_val->ToAbstract());
    auto getitem = node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fused_matmul, idx});
    (void)getitems->emplace_back(getitem);
  }
}
}  // namespace opt
}  // namespace mindspore
