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
#include <string>
#include "ir/meta_func_graph.h"
#include "ir/func_graph.h"

#include "include/common/pybind_api/api_register.h"

namespace mindspore {
py::dict UpdateFuncGraphHyperParams(const FuncGraphPtr &func_graph, const py::dict &params_init) {
  py::dict hyper_params;
  for (const auto &param : func_graph->parameters()) {
    auto param_node = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);
    py::str param_name = py::str(param_node->name());
    if (param_node->has_default()) {
      const char kModelName[] = "mindspore.common.parameter";
      const char kParamName[] = "Parameter";
      const char kParamInfoName[] = "ParamInfo";
      const py::module &mod = py::module::import(kModelName);
      const py::object &ParamInit = mod.attr(kParamName);
      const py::object &ParamInfoInit = mod.attr(kParamInfoName);
      const auto &old_value = param_node->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(old_value);
      py::object new_param;

      if (params_init.contains(param_name)) {
        const auto &new_value = params_init[param_name].cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(new_value);
        if (new_value->shape() != old_value->shape() || new_value->data_type() != old_value->data_type()) {
          MS_EXCEPTION(ValueError)
            << "Only support update parameter by Tensor or Parameter with same shape and dtype as it. "
               "The parameter '"
            << param_name.cast<std::string>() << "' has shape " << old_value->shape() << " and dtype "
            << TypeIdLabel(old_value->data_type()) << ", but got the update value with shape " << new_value->shape()
            << " and dtype " << TypeIdLabel(new_value->data_type()) << ".";
        }
        new_param = ParamInit(*new_value);
      } else {
        new_param = ParamInit(*old_value);
      }
      py::setattr(new_param, "param_info", ParamInfoInit(old_value->param_info()));
      param_node->set_default_param(new_param.cast<tensor::TensorPtr>());
      hyper_params[param_name] = new_param;
    }
  }
  return hyper_params;
}

void RegFuncGraph(const py::module *m) {
  // Define python "FuncGraph" class
  (void)py::class_<FuncGraph, FuncGraphPtr>(*m, "FuncGraph")
    .def(py::init())
    .def("str", &FuncGraph::ToString, "Get FuncGraph string representation.")
    .def("get_return", &FuncGraph::get_return, "Get return node of FuncGraph");
}

void RegMetaFuncGraph(const py::module *m) {
  // Define python "MetaFuncGraph_" class
  (void)py::class_<MetaFuncGraph, std::shared_ptr<MetaFuncGraph>>(*m, "MetaFuncGraph_")
    .def("set_signatures", &MetaFuncGraph::set_signatures, "Set primitive inputs signature.");
}

void RegUpdateFuncGraphHyperParams(py::module *m) {
  (void)m->def("update_func_graph_hyper_params", &UpdateFuncGraphHyperParams, py::arg("func_graph"),
               py::arg("params_init"), "Update FuncGraph hyper parameters, and return the updated parameters.");
}
}  // namespace mindspore
