/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/reshape_reshape_fusion.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <unordered_map>
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "ops/reshape.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
const auto kPreReshapePattern = "PreReshapePatternName";
const auto kPostReshapePattern = "PostReshapePatternName";
const auto kReshapeReshapePattern = "ReshapeReshapePatternName";
bool IsReshapeClusterOp(const BaseRef &n) {
  if (!utils::isa<AnfNodePtr>(n)) {
    return false;
  }
  auto anf_node = utils::cast<AnfNodePtr>(n);
  return CheckPrimitiveType(anf_node, prim::kPrimSqueeze) || CheckPrimitiveType(anf_node, prim::kPrimUnsqueeze);
}
}  // namespace

VectorRef ReshapeReshapeFusion::DefinePreReshapePattern() const {
  auto input_node = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_node != nullptr, {});
  auto reshape_shape = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(reshape_shape != nullptr, {});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto reshape = VectorRef({is_reshape, input_node, reshape_shape});
  auto is_reshape_cluster = std::make_shared<CondVar>(std::bind(IsReshapeClusterOp, p1));
  MS_CHECK_TRUE_RET(is_reshape_cluster != nullptr, {});
  auto is_seq = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq != nullptr, {});
  return VectorRef({is_reshape_cluster, reshape, is_seq});
}

VectorRef ReshapeReshapeFusion::DefinePostReshapePattern() const {
  // define pattern of reshape + squeeze/unsqueeze etc.
  auto input_node = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_node != nullptr, {});
  auto is_seq = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq != nullptr, {});
  auto is_reshape_cluster = std::make_shared<CondVar>(std::bind(IsReshapeClusterOp, p1));
  MS_CHECK_TRUE_RET(is_reshape_cluster != nullptr, {});
  auto reshape_cluster = VectorRef({is_reshape_cluster, input_node, is_seq});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto reshape_shape = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_shape != nullptr, {});
  return VectorRef({is_reshape, reshape_cluster, reshape_shape});
}

VectorRef ReshapeReshapeFusion::DefineReshapeReshapePattern() const {
  auto input_node = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_node != nullptr, {});
  // non-parameter could be supported, while a model convert failed without a known reason.
  auto reshape_shape = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(reshape_shape != nullptr, {});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto reshape = VectorRef({is_reshape, input_node, reshape_shape});
  auto is_reshape_cluster = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape_cluster != nullptr, {});
  auto shape_node = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(shape_node != nullptr, {});
  return VectorRef({is_reshape_cluster, reshape, shape_node});
}

std::unordered_map<std::string, VectorRef> ReshapeReshapeFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kPreReshapePattern] = DefinePreReshapePattern();
  patterns[kPostReshapePattern] = DefinePostReshapePattern();
  patterns[kReshapeReshapePattern] = DefineReshapeReshapePattern();
  return patterns;
}

AnfNodePtr GenerateNewShapeParam(const CNodePtr &cnode, const CNodePtr &reshape_cnode, const FuncGraphPtr &func_graph) {
  // calculate new shape for reshape.
  MS_ASSERT(cnode != nullptr && reshape_cnode != nullptr && func_graph != nullptr);
  MS_CHECK_TRUE_RET(reshape_cnode->size() == kInputSizeThree, nullptr);
  auto shape_node = reshape_cnode->input(kInputIndexTwo);
  MS_CHECK_TRUE_RET(shape_node != nullptr, nullptr);
  if (!IsParamNode(shape_node)) {
    return nullptr;
  }
  auto shape_param = shape_node->cast<ParameterPtr>();
  MS_CHECK_TRUE_RET(shape_param != nullptr, nullptr);
  lite::DataInfo data_info;
  (void)lite::FetchFromDefaultParam(shape_param, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_RET(data_info.shape_.size() == 1, nullptr);
  std::vector<int> shape(data_info.shape_.at(0));
  if (memcpy_s(shape.data(), shape.size() * sizeof(int), data_info.data_ptr_, shape.size() * sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "memcpy shape data failed.";
    return nullptr;
  }

  auto infer_squeeze = [](const std::vector<int> &in_shape, const std::vector<size_t> &axis) {
    auto sorted_axis = axis;
    std::sort(sorted_axis.begin(), sorted_axis.end(), std::greater<size_t>());
    std::vector<int> dst_shape = in_shape;
    for (auto dim : sorted_axis) {
      if (dim >= dst_shape.size() || dst_shape.at(dim) != 1) {
        return std::vector<int>();
      }
      (void)dst_shape.erase(dst_shape.begin() + dim);
    }
    return dst_shape;
  };
  auto infer_unsqueeze = [](const std::vector<int> &in_shape, const std::vector<size_t> &axis) {
    auto sorted_axis = axis;
    std::sort(sorted_axis.begin(), sorted_axis.end(), std::greater<size_t>());
    std::vector<int> dst_shape = in_shape;
    for (auto dim : sorted_axis) {
      if (dim > in_shape.size()) {
        return std::vector<int>();
      }
      (void)dst_shape.insert(dst_shape.begin() + dim, 1);
    }
    return dst_shape;
  };

  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto axis_attr = prim->GetAttr(ops::kAxis);
  MS_CHECK_TRUE_RET(axis_attr != nullptr, nullptr);
  auto axis_value = GetValue<std::vector<int64_t>>(axis_attr);
  std::vector<int> new_shape;
  if (axis_value.empty()) {
    (void)std::for_each(shape.begin(), shape.end(), [&new_shape](int ele) {
      if (ele != 1) new_shape.push_back(ele);
    });
  } else {
    std::vector<size_t> axis;
    (void)std::transform(axis_value.begin(), axis_value.end(), std::back_inserter(axis),
                         [&shape](int64_t x) { return x >= 0 ? x : x + shape.size(); });
    if (CheckPrimitiveType(cnode, prim::kPrimSqueeze)) {
      new_shape = infer_squeeze(shape, axis);
    } else if (CheckPrimitiveType(cnode, prim::kPrimUnsqueeze)) {
      new_shape = infer_unsqueeze(shape, axis);
    }
  }
  return opt::BuildIntVecParameterNode(func_graph, new_shape, "new_shape");
}

AnfNodePtr ReshapeReshapeFusion::Process(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                         const AnfNodePtr &node, const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cnode != nullptr && cnode->size() >= kInputSizeTwo, nullptr);
  auto pre_cnode = cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(pre_cnode != nullptr && pre_cnode->size() >= kInputSizeTwo, nullptr);
  if (opt::IsMultiOutputTensors(func_graph, pre_cnode)) {
    return nullptr;
  }

  auto input_node = pre_cnode->input(1);
  MS_CHECK_TRUE_RET(input_node != nullptr, nullptr);

  AnfNodePtr shape_node = nullptr;
  if (pattern_name == kPreReshapePattern) {
    shape_node = GenerateNewShapeParam(cnode, pre_cnode, func_graph);
  } else {
    MS_CHECK_TRUE_RET(cnode->size() == kInputSizeThree, nullptr);
    shape_node = cnode->input(kInputIndexTwo);
  }
  if (shape_node == nullptr) {
    MS_LOG(ERROR) << "Get shape node failed.";
    return nullptr;
  }

  // create new reshape op
  auto reshape_prim = std::make_shared<ops::Reshape>();
  if (reshape_prim == nullptr) {
    MS_LOG(ERROR) << "Build reshape primitive failed.";
    return nullptr;
  }
  auto reshape_prim_c = reshape_prim->GetPrim();
  MS_CHECK_TRUE_RET(reshape_prim_c != nullptr, nullptr);
  auto value_node = NewValueNode(reshape_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  auto new_reshape = func_graph->NewCNode({value_node, input_node, shape_node});
  MS_CHECK_TRUE_RET(new_reshape != nullptr, nullptr);
  new_reshape->set_fullname_with_scope(cnode->fullname_with_scope());
  return new_reshape;
}
}  // namespace mindspore::opt
