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

#include "tools/converter/adapter/acl/mapper/random_normal_mapper.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"
#include "ops/standard_normal.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kRandomNormalMaxInputSize = 2;
constexpr size_t kRandomNormalShapeLikeInputIndex = 1;
}  // namespace
STATUS RandomNormalMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "FuncGraph of node " << cnode->fullname_with_scope() << " is nullptr";
    return lite::RET_ERROR;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "FuncGraphManager of node " << cnode->fullname_with_scope() << " is nullptr";
    return lite::RET_ERROR;
  }
  ops::RandomNormal random_normal(src_prim);
  float mean = 0;
  float scale = 1;
  ops::StandardNormal standard_normal_op;
  if (random_normal.HasAttr(ops::kSeed)) {
    auto seed = random_normal.get_seed();
    standard_normal_op.set_seed(seed);
    standard_normal_op.set_seed2(seed);
  }
  if (random_normal.HasAttr(ops::kMean)) {
    mean = random_normal.get_mean();
  }
  if (random_normal.HasAttr(ops::kScale)) {
    scale = random_normal.get_scale();
  }
  TypeId type_id = kNumberTypeFloat32;
  if (src_prim->HasAttr(ops::kDataType)) {
    type_id = static_cast<TypeId>(GetValue<int64_t>(src_prim->GetAttr(ops::kDataType)));
  }
  if (cnode->size() > kRandomNormalShapeLikeInputIndex) {
    auto shape_like_node = cnode->input(kRandomNormalShapeLikeInputIndex);
    auto shape_node = NewCNode(cnode, prim::kPrimShape, {shape_like_node}, {}, kNumberTypeInt32,
                               cnode->fullname_with_scope() + "_shape");
    if (!shape_node) {
      MS_LOG(ERROR) << "Failed to create shape input for node " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    manager->SetEdge(cnode, kRandomNormalShapeLikeInputIndex, shape_node);
  } else if (src_prim->HasAttr(ops::kShape)) {
    auto shape = GetValue<ShapeVector>(src_prim->GetAttr(ops::kShape));
    std::vector<int32_t> shape_int32;
    std::transform(shape.begin(), shape.end(), std::back_inserter(shape_int32),
                   [](auto &dim) { return static_cast<int32_t>(dim); });
    auto shape_node = opt::BuildIntVecParameterNode(func_graph, shape_int32, cnode->fullname_with_scope() + "_shape");
    if (!shape_node) {
      MS_LOG(ERROR) << "Failed to create shape input for node " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    manager->AddEdge(cnode, shape_node);
  } else {
    MS_LOG(ERROR) << "RandomNormal node does not has attribute shape or shape input, node "
                  << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto dst_prim = standard_normal_op.GetPrim();
  dst_prim->set_attr(ops::kOutputDType, TypeIdToType(type_id));
  value_node->set_value(dst_prim);
  CNodePtr cur_node = cnode;
  if (scale != 1) {
    auto scale_param = opt::BuildFloatValueParameterNode(func_graph, scale, cnode->fullname_with_scope() + "_scale");
    if (scale_param == nullptr) {
      MS_LOG(ERROR) << "Failed to create scale parameter for node " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    auto mul_node = NewCNode(cnode, prim::kPrimMul, {cnode, scale_param}, cnode->abstract()->Clone(),
                             cnode->fullname_with_scope() + "_scale");
    if (mul_node == nullptr) {
      MS_LOG(ERROR) << "Failed to create scale node for node " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    cur_node = mul_node;
  }
  if (mean != 0) {
    auto mean_param = opt::BuildFloatValueParameterNode(func_graph, mean, cnode->fullname_with_scope() + "_mean");
    if (mean_param == nullptr) {
      MS_LOG(ERROR) << "Failed to create mean parameter of node " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    auto add_node = NewCNode(cnode, prim::kPrimAdd, {cnode, mean_param}, cnode->abstract()->Clone(),
                             cnode->fullname_with_scope() + "_mean");
    if (add_node == nullptr) {
      MS_LOG(ERROR) << "Failed to create mean node for node " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    cur_node = add_node;
  }
  if (cur_node != cnode) {
    manager->Replace(cnode, cur_node);
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameRandomNormal, RandomNormalMapper)
}  // namespace lite
}  // namespace mindspore
