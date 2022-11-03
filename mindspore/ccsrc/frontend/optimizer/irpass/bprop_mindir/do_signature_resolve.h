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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_DO_SIGNATURE_RESOLVE_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_DO_SIGNATURE_RESOLVE_H

#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class DoSignatureResolve : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    auto cnode = dyn_cast_ptr<CNode>(node);
    if (cnode == nullptr) {
      return nullptr;
    }
    auto prim_py = dyn_cast_ptr<PrimitivePy>(GetValueWithoutDoSignature(cnode->input(0)));
    if (prim_py == nullptr) {
      return nullptr;
    }
    auto prim_py_obj = prim_py->GetPyObj();
    // Resolve constexpr function
    if (py::hasattr(prim_py_obj, "fn")) {
      py::object fn = py::getattr(prim_py_obj, "fn");
      if (!fn || py::isinstance<py::none>(fn)) {
        return nullptr;
      }
      if (!py::hasattr(fn, "__module__") || !py::hasattr(fn, "__name__")) {
        return nullptr;
      }
      auto module = py::getattr(fn, "__module__");
      auto name = py::getattr(fn, "__name__");
      prim_py->AddAttr("constexpr_module", MakeValue(py::cast<std::string>(module)));
      prim_py->AddAttr("constexpr_name", MakeValue(py::cast<std::string>(name)));
      return nullptr;
    }
    // Convert the primitive with python infer or check function to class type
    if (py::hasattr(prim_py_obj, "__infer__") || py::hasattr(prim_py_obj, "__check__")) {
      auto class_type = py::str(py::getattr(prim_py_obj, "__class__")).cast<std::string>();
      auto mindir_class_type =
        NewValueNode(std::make_shared<MindIRClassType>(std::string(class_type.begin() + 1, class_type.end() - 1)));
      auto fg = node->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      const auto &inputs = cnode->inputs();
      std::vector<AnfNodePtr> new_inputs{fg->NewCNode({mindir_class_type})};
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(new_inputs));
      return fg->NewCNode(new_inputs);
    }
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_DO_SIGNATURE_RESOLVE_H
