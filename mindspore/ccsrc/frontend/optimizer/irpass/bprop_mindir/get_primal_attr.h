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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_PRIMAL_ATTR_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_PRIMAL_ATTR_H

#include <string>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class GetPrimalAttr : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimGetAttr, {IsVNode, IsVNode})(node);

    if (!is_match_) {
      return nullptr;
    }
    if (!prim_->isa<PrimitivePy>()) {
      return nullptr;
    }
    auto prim_py = prim_->cast_ptr<PrimitivePy>();
    MS_EXCEPTION_IF_NULL(prim_py);
    py::object attr_obj = py::getattr(prim_py->GetPyObj(), attr_.c_str());
    ValuePtr convert_result = nullptr;
    if (!parse::ConvertData(attr_obj, &convert_result)) {
      MS_LOG(EXCEPTION) << "Get the attr '" << attr_ << "' of python obj '" << py::str(prim_py->GetPyObj())
                        << "' failed.";
    }
    return NewValueNode(convert_result);
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<Primitive>(vnode)) {
      prim_ = GetValueNode<PrimitivePtr>(vnode);
    } else if (IsValueNode<StringImm>(vnode)) {
      attr_ = GetValueNode<StringImmPtr>(vnode)->value();
    }
    if (prim_ != nullptr && !attr_.empty()) {
      is_match_ = true;
    }
  }

  void Reset() {
    is_match_ = false;
    prim_ = nullptr;
    attr_.clear();
  }

 private:
  bool is_match_{false};
  PrimitivePtr prim_{nullptr};
  std::string attr_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_PRIMAL_ATTR_H
