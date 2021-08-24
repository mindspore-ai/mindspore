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
#include "backend/optimizer/pass/conv_transpose_to_conv_bp.h"
#include <memory>
#include <vector>
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCNodePrimitiveIdx = 0;
}  // namespace

const BaseRef ConvTransposeToConvBackpropInputPass::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto conv_transpose = std::make_shared<Primitive>(kConv2DTransposeOpName);
  return VectorRef({conv_transpose, Xs});
}

const AnfNodePtr ConvTransposeToConvBackpropInputPass::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto conv_transpose = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(conv_transpose);

  if (conv_transpose->size() == kCNodePrimitiveIdx) {
    MS_LOG(EXCEPTION) << "Invalid cnode " << node->DebugString() << " input size " << conv_transpose->size();
  }

  auto prim = GetValueNode<PrimitivePtr>(conv_transpose->input(kCNodePrimitiveIdx));
  MS_EXCEPTION_IF_NULL(prim);
  prim->Named::operator=(Named(kConv2DBackpropInputOpName));

  return node;
}
}  // namespace opt
}  // namespace mindspore
