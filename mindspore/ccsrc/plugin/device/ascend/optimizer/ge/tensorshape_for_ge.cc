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

#include "plugin/device/ascend/optimizer/ge/tensorshape_for_ge.h"
#include <vector>
#include <memory>
#include "include/common/utils/anfalgo.h"
#include "transform/graph_ir/transform_util.h"

namespace mindspore {
namespace opt {
namespace {
constexpr char kDtypeAttrName[] = "dtype";
}  // namespace

const BaseRef TensorShapeForGE::DefinePattern() const {
  VarPtr V = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

// Set the attr dtype and convert it to ge_dtype
const AnfNodePtr TensorShapeForGE::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  static const PrimitiveSet need_dtype_attr_nodes = {prim::kPrimShape, prim::kPrimTensorShape, prim::kPrimDynamicShape};
  PrimitivePtr prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  if (need_dtype_attr_nodes.find(prim) == need_dtype_attr_nodes.end()) {
    return nullptr;
  }

  if (common::AnfAlgo::HasNodeAttr(kDtypeAttrName, cnode)) {
    return nullptr;
  }

  // get output dtype (ms_dtype)
  TypeId output_dtype = common::AnfAlgo::GetOutputInferDataType(cnode, 0);
  // convert to ge_dtype
  int64_t ge_dtype = static_cast<int64_t>(transform::TransformUtil::ConvertDataType(output_dtype));
  // update/set attr
  common::AnfAlgo::SetNodeAttr(kDtypeAttrName, MakeValue(ge_dtype), cnode);

  return node;
}
}  // namespace opt
}  // namespace mindspore
