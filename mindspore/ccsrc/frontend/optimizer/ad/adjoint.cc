/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/ad/adjoint.h"

#include <utility>

#include "ir/anf.h"
#include "frontend/optimizer/ad/dfunctor.h"

namespace mindspore {
namespace ad {
Adjoint::Adjoint(const AnfNodePtr &primal, const AnfNodePtr &k, const FuncGraphPtr &caller)
    : primal_(primal), caller_(caller), dout_(nullptr) {
  if (k != nullptr) {
    k_ = k;
    MS_LOG(DEBUG) << "Add adjoint for " << primal->ToString() << " " << k_->ToString();
  } else {
    // Init k hole in a recursive case.
    auto k_hole = std::make_shared<Primitive>("k_hole");
    (void)k_hole->AddAttr("info", MakeValue(primal->ToString()));
    k_ = NewValueNode(k_hole);
    MS_LOG(DEBUG) << "Add hole for " << primal->ToString() << " " << k_->ToString();
  }

  dout_hole_ = caller_->NewCNodeInFront({NewValueNode(prim::GetPythonOps("zeros_like")), k_});
  RegisterKUser(dout_hole_->cast<CNodePtr>(), 1);
}

AnfNodePtr Adjoint::k() { return k_; }

void Adjoint::RegisterKUser(const CNodePtr &user, size_t index) { k_user_.emplace_back(std::make_pair(user, index)); }

void Adjoint::UpdateK(const AnfNodePtr &new_k) {
  MS_EXCEPTION_IF_NULL(new_k);
  MS_LOG(DEBUG) << "Replace k " << k_->ToString() << " with " << new_k->ToString();
  // In recursive case, it needs update.
  for (auto &user : k_user_) {
    MS_LOG(DEBUG) << "Update k user " << user.first->ToString() << " " << user.second << " input with new_k"
                  << new_k->ToString();
    if (user.first->input(user.second) != k_) {
      MS_LOG(EXCEPTION) << "Update k user " << user.first->ToString() << " " << user.second << " input with new_k "
                        << new_k->ToString() << ", user relation is set wrongly";
    }
    user.first->set_input(user.second, new_k);
  }
  k_ = new_k;
}

AnfNodePtr Adjoint::primal() { return primal_; }

AnfNodePtr Adjoint::dout() { return dout_hole_; }

void Adjoint::RegisterDoutUser(const CNodePtr &user, size_t index) {
  dout_user_.emplace_back(std::make_pair(user, index));
}

void Adjoint::AccumulateDout(const AnfNodePtr &dout_factor) {
  if (dout_ != nullptr) {
    MS_LOG(DEBUG) << "Update dout " << dout_->ToString() << " with dout_factor " << dout_factor->ToString();
    auto add = prim::GetPythonOps("hyper_add");
    dout_ = caller_->NewCNode({NewValueNode(add), dout_, dout_factor});
    return;
  }
  dout_ = dout_factor;
}

void Adjoint::CallDoutHole() {
  if (dout_ != nullptr) {
    for (auto &user : dout_user_) {
      MS_LOG(DEBUG) << "Update dout user " << user.first->ToString() << " " << user.second << " input with dout "
                    << dout_->ToString();
      if (user.first->input(user.second) != dout_hole_) {
        MS_LOG(EXCEPTION) << "Update dout user " << user.first->ToString() << " " << user.second << " input with dout "
                          << dout_->ToString() << ", user relation is set wrongly";
      }
      user.first->set_input(user.second, dout_);
    }
  }
}
}  // namespace ad
}  // namespace mindspore
