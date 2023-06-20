/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/squeeze_expanddims_fusion.h"
#include <vector>
#include "mindspore/core/ops/array_ops.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ops/op_utils.h"
#include "ops/squeeze.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "include/registry/converter_context.h"

namespace mindspore::opt {
const BaseRef SqueezeExpandDimsFusion::DefinePattern() const {
  auto is_expanddims = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimExpandDims>);
  MS_CHECK_TRUE_RET(is_expanddims != nullptr, {});
  auto ex_shape = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(ex_shape != nullptr, {});
  auto is_squeeze = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSqueeze>);
  MS_CHECK_TRUE_RET(is_squeeze != nullptr, {});
  return VectorRef({is_expanddims, is_squeeze, ex_shape});
}

bool SqueezeExpandDimsFusion::CheckCanFuse(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto expanddims_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(expanddims_cnode != nullptr, false);
  MS_CHECK_TRUE_RET(expanddims_cnode->input(SECOND_INPUT) != nullptr, false);
  auto squeeze_cnode = expanddims_cnode->input(SECOND_INPUT)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(squeeze_cnode != nullptr, false);
  if (IsMultiOutputTensors(func_graph, squeeze_cnode)) {
    return false;
  }
  auto squeeze_primitive = GetCNodePrimitive(squeeze_cnode);
  MS_CHECK_TRUE_RET(squeeze_primitive != nullptr, false);
  MS_CHECK_TRUE_RET(!IsQuantParameterNode(squeeze_primitive), false);

  MS_CHECK_TRUE_RET(expanddims_cnode->input(THIRD_INPUT) != nullptr, false);
  lite::DataInfo data_info;
  if (lite::FetchConstData(expanddims_cnode, THIRD_INPUT, converter::kFmkTypeMs, &data_info, false) != lite::RET_OK) {
    return false;
  }
  if ((data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32) ||
      data_info.data_.size() != C4NUM) {
    return false;
  }
  auto expanddims_axis = *reinterpret_cast<int *>(data_info.data_.data());

  auto squeeze_prim = api::MakeShared<mindspore::ops::Squeeze>(squeeze_primitive);
  MS_CHECK_TRUE_RET(squeeze_prim != nullptr, false);
  auto squeeze_axises = squeeze_prim->get_axis();
  MS_CHECK_TRUE_RET(squeeze_axises.size() < DIMENSION_2D, false);
  int64_t squeeze_axis;
  if (squeeze_axises.size() < DIMENSION_1D) {
    squeeze_axis = C0NUM;
  } else {
    squeeze_axis = squeeze_axises.at(C0NUM);
  }
  if (squeeze_axis == expanddims_axis) {
    return true;
  } else {
    // squeeze_axis or expanddims_axis is less than zero
    MS_CHECK_TRUE_RET(squeeze_cnode->input(SECOND_INPUT) != nullptr, false);
    auto squeeze_abt = squeeze_cnode->input(SECOND_INPUT)->abstract();
    MS_CHECK_TRUE_RET(squeeze_abt != nullptr, false);
    std::vector<int64_t> squeeze_shape;
    if (FetchShapeFromAbstract(squeeze_abt, &squeeze_shape) != lite::RET_OK) {
      return false;
    }

    auto expanddims_prim = GetCNodePrimitive(expanddims_cnode);
    MS_CHECK_TRUE_RET(expanddims_prim != nullptr, false);
    auto is_inferred =
      expanddims_prim->GetAttr(kInferDone) != nullptr && GetValue<bool>(expanddims_prim->GetAttr(kInferDone));
    MS_CHECK_TRUE_RET(is_inferred, false);
    auto expanddims_abt = expanddims_cnode->abstract();
    MS_CHECK_TRUE_RET(expanddims_abt != nullptr, false);
    std::vector<int64_t> expanddims_shape;
    if (FetchShapeFromAbstract(expanddims_abt, &expanddims_shape) != lite::RET_OK) {
      return false;
    }
    MS_CHECK_TRUE_RET(squeeze_shape == expanddims_shape, false);
  }
  return true;
}

const AnfNodePtr SqueezeExpandDimsFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }

  if (!CheckCanFuse(func_graph, node)) {
    return nullptr;
  }

  auto expanddims_cnode = node->cast<CNodePtr>();
  auto squeeze_cnode = expanddims_cnode->input(SECOND_INPUT)->cast<CNodePtr>();
  auto manage = Manage(func_graph);
  MS_CHECK_TRUE_RET(manage != nullptr, nullptr);
  manage->Replace(expanddims_cnode, squeeze_cnode->input(SECOND_INPUT));
  return nullptr;
}
}  // namespace mindspore::opt
