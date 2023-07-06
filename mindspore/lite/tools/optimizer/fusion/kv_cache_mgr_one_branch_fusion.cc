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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/kv_cache_mgr_one_branch_fusion.h"
#include <memory>
#include <vector>
#include "schema/inner/model_generated.h"
#include "ops/affine.h"
#include "src/common/log_adapter.h"
#include "ops/splice.h"
#include "ops/mat_mul.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/array_ops.h"
#include "ops/math_ops.h"
#include "ops/comparison_ops.h"
#include "ops/nn_optimizer_ops.h"
#include "ops/fusion/kv_cache_mgr.h"
#include "ops/add.h"
#include "ops/expand_dims.h"
#include "ops/mul.h"
#include "ops/make_tuple.h"
#include "ops/concat.h"
#include "ops/assign.h"
namespace mindspore::opt {
const BaseRef KVCacheMgrOneBranchFusion::DefinePattern() const {
  if (!InitVar()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }
  // main branch
  auto is_reshape = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  VectorRef reshape_ref({is_reshape, input_0_batch_valid_length_, std::make_shared<Var>()});

  auto is_equal = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimEqual>);
  MS_CHECK_TRUE_RET(is_equal != nullptr, {});
  VectorRef equal_ref({is_equal, std::make_shared<Var>(), reshape_ref});

  auto is_cast = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast != nullptr, {});
  VectorRef cast_ref({is_cast, equal_ref, std::make_shared<Var>()});

  auto is_expandims = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimExpandDims>);
  MS_CHECK_TRUE_RET(is_expandims != nullptr, {});
  VectorRef expandims_ref({is_expandims, cast_ref, std::make_shared<Var>()});

  return expandims_ref;
}

tensor::TensorPtr KVCacheMgrOneBranchFusion::ConstData(int32_t padding_length) const {
  std::vector<int64_t> shp = {padding_length};
  tensor::TensorPtr const_data = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  auto *val = static_cast<int32_t *>(const_data->data_c());
  for (int i = 0; i < padding_length; ++i) {
    *(val + i) = 0;
  }
  return const_data;
}

bool KVCacheMgrOneBranchFusion::OutputIsMulAdd(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_user = manager->node_users()[node];
  const size_t mini_size = 2;
  if (node_user.size() < mini_size) {
    return false;
  }
  for (auto &user : node_user) {
    auto mul_node = user.first;
    if (!utils::isa<CNode>(mul_node)) {
      return false;
    }
    if (!CheckPrimitiveType(mul_node, prim::kPrimMul)) {
      return false;
    }
    auto mul_node_user = manager->node_users()[mul_node];
    if (mul_node_user.size() > 1) {
      return false;
    }
    auto add_node = mul_node_user.front().first;
    if (!utils::isa<CNode>(add_node)) {
      return false;
    }
    if (!CheckPrimitiveType(add_node, prim::kPrimAdd)) {
      return false;
    }
  }
  return true;
}

CNodePtr KVCacheMgrOneBranchFusion::CreateConcatNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  auto make_tuple_prim = std::make_shared<ops::MakeTuple>();
  MS_CHECK_TRUE_RET(make_tuple_prim != nullptr, nullptr);
  auto make_tuple_prim_c = make_tuple_prim->GetPrim();
  MS_CHECK_TRUE_RET(make_tuple_prim_c != nullptr, nullptr);

  auto input_0_batch_valid_length_node = utils::cast<AnfNodePtr>((*equiv)[input_0_batch_valid_length_]);
  MS_ASSERT(input_0_batch_valid_length_node != nullptr);

  auto batch_abstruct = input_0_batch_valid_length_node->abstract();
  MS_CHECK_TRUE_RET(batch_abstruct != nullptr, nullptr);

  auto batch_shape = batch_abstruct->BuildShape();
  MS_EXCEPTION_IF_NULL(batch_shape);
  auto batch_shape_ptr = dyn_cast<abstract::Shape>(batch_shape);
  MS_EXCEPTION_IF_NULL(batch_shape_ptr);
  ShapeVector batch_shape_vec = batch_shape_ptr->shape();
  const int total_len = ((batch_shape_vec.size() + 7) / 8) * 8;
  const int padding_len = total_len - batch_shape_vec.size();

  auto padding_tensor = ConstData(padding_len);
  auto padding_value_node = NewValueNode(padding_tensor);
  padding_value_node->set_abstract(padding_tensor->ToAbstract());
  func_graph->AddValueNode(padding_value_node);

  auto make_tuple_cnode =
    func_graph->NewCNode(make_tuple_prim_c, {input_0_batch_valid_length_node, padding_value_node});
  AbstractBasePtrList abstract_list;
  (void)abstract_list.emplace_back(input_0_batch_valid_length_node->abstract());
  (void)abstract_list.emplace_back(padding_value_node->abstract());
  make_tuple_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));

  auto concat_prim = std::make_shared<ops::Concat>();
  MS_CHECK_TRUE_RET(concat_prim != nullptr, nullptr);
  concat_prim->set_axis(0);
  const int input_num = 2;
  (void)concat_prim->AddAttr("N", api::MakeValue(input_num));
  (void)concat_prim->AddAttr("inputNums", api::MakeValue(input_num));
  auto concat_prim_c = concat_prim->GetPrim();
  MS_CHECK_TRUE_RET(concat_prim_c != nullptr, nullptr);

  ShapeVector concat_shape = {total_len};
  auto shape_ptr = std::make_shared<abstract::Shape>(concat_shape);
  auto concat_cnode = func_graph->NewCNode(concat_prim_c, {make_tuple_cnode});
  concat_cnode->set_abstract(padding_value_node->abstract()->Clone());
  concat_cnode->abstract()->set_shape(shape_ptr);

  return concat_cnode;
}

bool KVCacheMgrOneBranchFusion::InitVar() const {
  input_0_batch_valid_length_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_0_batch_valid_length_ != nullptr, false);
  return true;
}

const AnfNodePtr KVCacheMgrOneBranchFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    return nullptr;
  }

  if (!OutputIsMulAdd(func_graph, node)) {
    MS_LOG(INFO) << "Not is KVCache Pattern.";
    return nullptr;
  }

  auto cnode = CreateConcatNode(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "new concat node failed.";
    return nullptr;
  }
  return cnode;
}
}  // namespace mindspore::opt
