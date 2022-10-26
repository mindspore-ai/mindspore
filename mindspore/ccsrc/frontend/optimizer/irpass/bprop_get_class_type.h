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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_GET_CLASS_TYPE_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_GET_CLASS_TYPE_H

#include <string>
#include <memory>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class BpropGetClassType : public AnfVisitor {
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsValueNode<MindIRClassType>(node)) {
      return nullptr;
    }
    auto class_path = GetValueNode<MindIRClassTypePtr>(node)->name();
    std::string sub_str = ".";
    auto class_name_pos =
      std::find_end(class_path.begin(), class_path.end(), sub_str.begin(), sub_str.end()) - class_path.begin();
    auto package = std::string(class_path.begin(), class_path.begin() + class_name_pos);
    auto class_name = std::string(class_path.begin() + class_name_pos + 1, class_path.end());
    auto module = python_adapter::GetPyModule(package);
    if (!module || py::isinstance<py::none>(module)) {
      MS_LOG(EXCEPTION) << "Can not get python module: " << package;
    }
    auto attr = module.attr(class_name.c_str());
    return NewValueNode(std::make_shared<parse::ClassType>(attr, class_path));
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_GET_CLASS_TYPE_H
