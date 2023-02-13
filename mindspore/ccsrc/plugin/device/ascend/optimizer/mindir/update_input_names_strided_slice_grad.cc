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
#include "plugin/device/ascend/optimizer/mindir/update_input_names_strided_slice_grad.h"
#include <memory>
#include <vector>
#include <string>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kXs = "Xs";
constexpr auto kMSliceGrad = "m_slice_grad";
constexpr auto kRSliceGrad = "r_slice_grad";

AnfNodePtr BuildSliceGrad(const PatternMap &m, const AnfNodePtr &) {
  auto node = m.Get(kMSliceGrad);
  MS_EXCEPTION_IF_NULL(node);
  auto strided_slice_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(strided_slice_grad);

  const size_t shapex_index = 1;
  auto primitive = common::AnfAlgo::GetCNodePrimitive(strided_slice_grad);
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_names_ptr = primitive->GetAttr(kAttrInputNames);
  MS_EXCEPTION_IF_NULL(input_names_ptr);
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names_ptr);
  input_names_vec[shapex_index] = "shape";
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names_vec), strided_slice_grad);
  return strided_slice_grad;
}
}  // namespace

bool StridedSliceGradUpdateInputNames::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &,
                                                       const AnfNodePtr &node) const {
  auto strided_slice_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(strided_slice_grad);
  if (common::AnfAlgo::IsDynamicShape(strided_slice_grad)) {
    return true;
  }
  return false;
}

void StridedSliceGradUpdateInputNames::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddSeqVar(kXs).AddCNode(kMSliceGrad, {std::make_shared<Primitive>(kStridedSliceGradOpName), kXs});
}

void StridedSliceGradUpdateInputNames::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(kRSliceGrad, {std::make_shared<Primitive>(kStridedSliceGradOpName), kXs}, BuildSliceGrad);
}
}  // namespace opt
}  // namespace mindspore
