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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_CONSTEXPR_OPS_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_CONSTEXPR_OPS_H

#include <string>
#include <memory>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class GetConstexprOps : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto prim = dyn_cast_ptr<Primitive>(GetValueWithoutDoSignature(node));
    if (prim == nullptr) {
      return nullptr;
    }
    auto module_value = prim->GetAttr("constexpr_module");
    if (module_value == nullptr) {
      return nullptr;
    }
    auto package = GetValue<std::string>(module_value);
    auto class_name = GetValue<std::string>(prim->GetAttr("constexpr_name"));
    auto module = python_adapter::GetPyModule(package);
    if (!module || py::isinstance<py::none>(module)) {
      MS_LOG(EXCEPTION) << "Can not get python module: " << package;
    }
    auto attr = module.attr(class_name.c_str());
    auto prim_adapter = attr.cast<PrimitivePyAdapterPtr>();
    MS_EXCEPTION_IF_NULL(prim_adapter);
    auto new_prim = prim_adapter->attached_primitive();
    if (new_prim == nullptr) {
      new_prim = std::make_shared<PrimitivePy>(attr, prim_adapter);
      prim_adapter->set_attached_primitive(new_prim);
    }
    return NewValueNode(std::make_shared<prim::DoSignaturePrimitive>(new_prim->name(), new_prim));
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_CONSTEXPR_OPS_H
