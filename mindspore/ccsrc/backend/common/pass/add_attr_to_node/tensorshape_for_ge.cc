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

#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr char kDtypeAttrName[] = "dtype";
constexpr char kDisableKernelBackoff[] = "MS_DISABLE_KERNEL_BACKOFF";
}  // namespace

const AnfNodePtr TensorShapeAddDtype(const FuncGraphPtr &, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (common::AnfAlgo::HasNodeAttr(kDtypeAttrName, cnode)) {
    return nullptr;
  }

  // get output dtype (ms_dtype)
  TypeId output_dtype = common::AnfAlgo::GetOutputInferDataType(cnode, 0);
  // update/set attr
  common::AnfAlgo::SetNodeAttr(kDtypeAttrName, MakeValue(static_cast<int64_t>(output_dtype)), cnode);

  return node;
}

const AnfNodePtr AddOnlyDependShapeAttr(const FuncGraphPtr &, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::GetEnv(kDisableKernelBackoff) == "1") {
    return node;
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrOnlyDependShape, cnode)) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrOnlyDependShape, MakeValue(std::vector<bool>{true}), node);
  return node;
}

const AnfNodePtr TensorShapeProcess(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  auto ret1 = TensorShapeAddDtype(fg, node);
  auto ret2 = AddOnlyDependShapeAttr(fg, node);
  return (ret1 != nullptr || ret2 != nullptr) ? node : nullptr;
}
}  // namespace opt
}  // namespace mindspore
