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
#include <iostream>
#include "common/common_test.h"
#include "transform/transform_base_test.h"
#include "pybind_api/ir/tensor_py.h"
#include "pipeline/jit/parse/resolve.h"

using mindspore::tensor::TensorPy;

namespace mindspore {
namespace transform {
using mindspore::parse::ParsePythonCode;
namespace python_adapter = mindspore::parse::python_adapter;
using mindspore::parse::ResolveAll;
std::vector<FuncGraphPtr> getAnfGraph(string package, string function) {
  py::function fn_ = python_adapter::GetPyFn(package, function);
  FuncGraphPtr func_graph = ParsePythonCode(fn_);
  std::vector<FuncGraphPtr> graphVector;
  graphVector.clear();
  if (nullptr == func_graph) return graphVector;

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  if (!ret_) return graphVector;

  // get  graph
  for (auto func_graph : manager->func_graphs()) {
    graphVector.push_back(func_graph);
  }
  return graphVector;
}

void PrintMeTensor(MeTensor* tensor) {
#define DTYPE float
  DTYPE* me_data = reinterpret_cast<DTYPE*>((*tensor).data_c());
  size_t elements = (*tensor).ElementsNum();
  std::cout << "the in memory block data size is: " << std::dec << tensor->data().nbytes() << " bytes" << std::endl;
  std::cout << "the in memory block data is: " << std::endl;
  for (int i = 0; i < elements; i++) {
    std::cout << static_cast<DTYPE>(*(me_data + i)) << std::endl;
  }

  std::cout << "the py::str() data is: " << std::endl;
  py::array tensor_data = TensorPy::AsNumpy(*tensor);
  std::cout << std::string(py::str(tensor_data)) << std::endl;

  std::cout << "tensor dtype is: " << py::str(tensor_data.dtype()) << std::endl;
}

FuncGraphPtr MakeFuncGraph(const PrimitivePtr prim, unsigned int nparam) {
  // build the func_graph manually, eg:
  // MakeFuncGraph(std::make_shared<Primitive>("scalar_add"), 2) means:
  /* python source code:
   * @mindspore
   * def f(x, y):
   *     return x + y
   */
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim));
  for (unsigned int i = 0; i < nparam; i++) {
    if ((prim->name() == "ScalarSummary" || prim->name() == "TensorSummary" ||
        prim->name() == "ImageSummary" || prim->name() == "HistogramSummary") &&
        i == 0) {
      auto input = NewValueNode("testSummary");
      inputs.push_back(input);
    } else {
      auto input = func_graph->add_parameter();
      input->set_name("x" + std::to_string(i));
      inputs.push_back(input);
    }
  }
  CNodePtr cnode_prim = func_graph->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(std::make_shared<Primitive>("Return")));
  inputs.push_back(cnode_prim);
  CNodePtr cnode_return = func_graph->NewCNode(inputs);
  func_graph->set_return(cnode_return);
  return func_graph;
}

MeTensorPtr MakeTensor(const TypePtr& t, std::initializer_list<int64_t> shp) {
  auto shape = std::vector<int64_t>(shp);
  auto tensor = std::make_shared<tensor::Tensor>(t->type_id(), shape);
  return tensor;
}

}  // namespace transform
}  // namespace mindspore
