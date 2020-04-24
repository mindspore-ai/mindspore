/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/validator.h"

#include <memory>
#include <mutex>

#include "ir/manager.h"
#include "ir/dtype.h"
#include "./common.h"
#include "pipeline/static_analysis/prim.h"

namespace mindspore {
namespace validator {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractClass;
using mindspore::abstract::AbstractError;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractJTagged;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractType;

void ValidateOperation(const AnfNodePtr &node) {
  if (!IsValueNode<Primitive>(node)) {
    return;
  }

  // Primitive must in whitelist
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
  if (abstract::IsInWhiteList(prim)) {
    return;
  }
  if (prim->HasPyEvaluator()) {
    MS_LOG(DEBUG) << "Primitive " << prim->name() << " has python evaluator.";
    return;
  }
  if (prim->name() == "fake_bprop") {
    MS_LOG(EXCEPTION) << "Illegal primitive: " << GetValue<std::string>(prim->GetAttr("info"));
  }

  MS_LOG(EXCEPTION) << "Illegal primitive: " << prim->name();
}

void ValidateAbstract(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(WARNING) << "Node to validate is invalid";
    return;
  }
  AbstractBasePtr ptrBase = node->abstract();
  if (ptrBase == nullptr) {
    MS_LOG(WARNING) << "Abstract is null in node: " << node->DebugString();
    return;
  }
  if (ptrBase->isa<AbstractClass>() || ptrBase->isa<AbstractJTagged>()) {
    // Validate a type.
    MS_LOG(EXCEPTION) << "Illegal type in the graph: " << ptrBase->ToString();
  }
  if (ptrBase->isa<AbstractScalar>()) {
    TypePtr ptrType = ptrBase->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(ptrType);
    if (ptrType->isa<Problem>() || ptrType->isa<External>()) {
      // only send string in external
      if (!IsValueNode<StringImm>(node)) {
        // Validate a type.
        MS_LOG(EXCEPTION) << "Illegal type in the graph: " << ptrBase->ToString();
      }
    }
    return;
  }
  if (ptrBase->isa<AbstractError>()) {
    // NOTICE: validate dead code?
    MS_LOG(WARNING) << "AbstractError in the graph: " << ptrBase->ToString();
    return;
  }

  if (ptrBase->isa<AbstractType>() || ptrBase->isa<AbstractFunction>() || ptrBase->isa<AbstractTuple>() ||
      ptrBase->isa<AbstractList>() || ptrBase->isa<AbstractTensor>() || ptrBase->isa<abstract::AbstractRefKey>()) {
    return;
  }

  if (ptrBase->isa<abstract::AbstractNone>()) {
    return;
  }

  // Other types show exception
  MS_LOG(EXCEPTION) << "Illegal type in the graph: " << ptrBase->ToString();
}

void Validate(const FuncGraphPtr &fg) {
  FuncGraphManagerPtr mgr = Manage(fg, false);
  MS_EXCEPTION_IF_NULL(mgr);
  AnfNodeSet &all_nodes = mgr->all_nodes();
  for (const auto &anf_node : all_nodes) {
    ValidateOperation(anf_node);
    ValidateAbstract(anf_node);
  }
}
}  // namespace validator
}  // namespace mindspore
