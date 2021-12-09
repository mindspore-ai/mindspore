/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/dpico_preprocess_pass.h"
#include "include/registry/pass_registry.h"
#include "ops/op_utils.h"
#include "common/anf_util.h"
#include "common/op_enum.h"
#include "common/data_transpose_utils.h"
#include "ops/fusion/add_fusion.h"
#include "common/op_attr.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS InsertTransposeBeforeBiasAdd(const api::FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                    const ShapeVector &shape_vector, const abstract::AbstractBasePtr &abstract) {
  auto manager = func_graph->get_manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr. ";
    return RET_ERROR;
  }
  auto pre_trans_cnode =
    GenTransposeNode(func_graph, cnode->input(1), kNC2NH, cnode->fullname_with_scope() + "_converted_to_add_pre_0");
  if (pre_trans_cnode == nullptr) {
    MS_LOG(ERROR) << "create pre transpose cnode failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto pre_trans_abstract = abstract->Clone();
  if (pre_trans_abstract == nullptr) {
    MS_LOG(ERROR) << "create tensor abstract failed. ";
    return RET_ERROR;
  }
  ShapeVector nc2nh_shape{shape_vector.at(0), shape_vector.at(kInputIndex2), shape_vector.at(kInputIndex3),
                          shape_vector.at(1)};
  auto nc2nh_shape_ptr = std::make_shared<abstract::Shape>(nc2nh_shape);
  if (nc2nh_shape_ptr == nullptr) {
    MS_LOG(ERROR) << "new abstract shape failed.";
    return RET_ERROR;
  }
  pre_trans_abstract->set_shape(nc2nh_shape_ptr);
  pre_trans_cnode->set_abstract(pre_trans_abstract);
  auto pre_trans_prim = GetValueNode<PrimitivePtr>(pre_trans_cnode->input(0));
  MS_ASSERT(pre_trans_prim != nullptr);
  pre_trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NCHW));
  pre_trans_prim->AddAttr(kInferDone, MakeValue<bool>(true));
  manager->SetEdge(cnode, kInputIndex1, pre_trans_cnode);
  return RET_OK;
}
STATUS ReplaceBiasAddWithAdd(const api::FuncGraphPtr &func_graph, const CNodePtr &cnode,
                             const PrimitivePtr &primitive) {
  auto manager = func_graph->get_manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr. ";
    return RET_ERROR;
  }
  auto prim = std::make_shared<ops::AddFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new AddFusion failed." << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  prim->SetAttrs(primitive->attrs());
  prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
  auto add_value_node = NewValueNode(prim);
  if (add_value_node == nullptr) {
    MS_LOG(ERROR) << "new value node failed.";
    return RET_ERROR;
  }
  (void)manager->Replace(cnode->input(0), add_value_node);
  auto pre_trans_abstract = GetCNodeInputAbstract(cnode, kInputIndex1);
  if (pre_trans_abstract == nullptr) {
    MS_LOG(ERROR) << "cnode input_1 's abstract is nullptr. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  cnode->set_abstract(pre_trans_abstract->Clone());
  cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_converted_to_add");
  return RET_OK;
}
STATUS InsertTransposeAfterBiasAdd(const api::FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                   const ShapeVector &shape_vector) {
  auto manager = func_graph->get_manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr. ";
    return RET_ERROR;
  }
  auto post_trans_cnode = GenTransposeNode(func_graph, cnode, kNH2NC, cnode->fullname_with_scope() + "_post_0");
  if (post_trans_cnode == nullptr) {
    MS_LOG(ERROR) << "create post transpose cnode failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto post_trans_abstract = CreateTensorAbstract(shape_vector, kNumberTypeFloat32);
  if (post_trans_abstract == nullptr) {
    MS_LOG(ERROR) << "create tensor abstract failed. ";
    return RET_ERROR;
  }
  post_trans_cnode->set_abstract(post_trans_abstract->Clone());
  auto post_trans_prim = GetValueNode<PrimitivePtr>(post_trans_cnode->input(0));
  MS_ASSERT(post_trans_prim != nullptr);
  post_trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
  post_trans_prim->AddAttr(kInferDone, MakeValue<bool>(true));
  if (!manager->Replace(cnode, post_trans_cnode)) {
    MS_LOG(ERROR) << "replace biasadd with add failed." << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace
STATUS DpicoPreprocessPass::PreProcessBiadAdd(const api::FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto manager = func_graph->get_manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr. ";
    return RET_ERROR;
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr:" << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto abstract = GetCNodeInputAbstract(cnode, kInputIndex1);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "abstract is nullptr. " << cnode->input(1)->fullname_with_scope();
    return RET_ERROR;
  }
  ShapeVector shape_vector;
  if (FetchShapeFromAbstract(abstract, &shape_vector) != RET_OK) {
    MS_LOG(ERROR) << "fetch shape from abstract failed. " << cnode->input(1)->fullname_with_scope();
    return RET_ERROR;
  }
  if (shape_vector.empty() || shape_vector.size() != kDims4) {
    MS_LOG(DEBUG) << "shape is empty or shape vector size is not equal to 4 dims, don't need to insert transpose op.";
    return RET_NO_CHANGE;
  }

  if (InsertTransposeBeforeBiasAdd(func_graph, cnode, shape_vector, abstract) != RET_OK) {
    MS_LOG(ERROR) << "insert transpose before BiasAdd failed." << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (ReplaceBiasAddWithAdd(func_graph, cnode, primitive) != RET_OK) {
    MS_LOG(ERROR) << "replace BiasAdd with Add failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (InsertTransposeAfterBiasAdd(func_graph, cnode, shape_vector) != RET_OK) {
    MS_LOG(ERROR) << "insert transpose after BiasAdd failed." << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  return RET_OK;
}

bool DpicoPreprocessPass::Execute(const api::FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr.";
    return false;
  }

  auto manager = api::FuncGraphManager::Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }

  auto node_list = api::FuncGraph::TopoSort(func_graph->get_return());
  int status;
  for (const auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (CheckPrimitiveType(cnode, prim::kPrimBiasAdd)) {
      status = PreProcessBiadAdd(func_graph, cnode);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "preprocess biasadd for dpico failed.";
        return false;
      }
    }
  }
  return true;
}
}  // namespace dpico
}  // namespace mindspore
namespace mindspore::registry {
REG_PASS(DpicoPreprocessPass, dpico::DpicoPreprocessPass)
}
