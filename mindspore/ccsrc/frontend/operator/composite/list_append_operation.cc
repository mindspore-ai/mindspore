/**
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

#include "frontend/operator/composite/list_append_operation.h"

#include <vector>
#include <string>
#include <memory>

#include "abstract/param_validator.h"
#include "frontend/optimizer/opt.h"
#include "pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
FuncGraphPtr ListAppend::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  abstract::CheckArgsSize("ListAppend", args_list, 2);

  AbstractBasePtr arg0 = args_list[0];
  abstract::AbstractListPtr arg0_list = dyn_cast<abstract::AbstractList>(arg0);
  MS_EXCEPTION_IF_NULL(arg0_list);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("append");
  AnfNodePtr arg0_node = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeList));
  size_t arg0_length = arg0_list->size();
  for (size_t i = 0; i < arg0_length; ++i) {
    elems.push_back(ret->NewCNode({NewValueNode(prim::kPrimListGetItem), arg0_node, NewValueNode(SizeToLong(i))}));
  }
  AnfNodePtr arg1_node = ret->add_parameter();
  elems.push_back(arg1_node);

  ret->set_output(ret->NewCNode(elems));
  return ret;
}

REGISTER_PYBIND_DEFINE(ListAppend_, ([](const py::module *m) {
                         (void)py::class_<ListAppend, MetaFuncGraph, std::shared_ptr<ListAppend>>(*m, "ListAppend_")
                           .def(py::init<std::string &>());
                       }));
}  // namespace prim
}  // namespace mindspore
