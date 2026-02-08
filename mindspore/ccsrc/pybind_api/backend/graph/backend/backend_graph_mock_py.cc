/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mindspore/core/include/ir/value.h"
#include "mindspore/core/include/ir/primitive.h"
#include "mindspore/core/include/ir/graph_utils.h"
#include "mindspore/core/include/ir/func_graph.h"
#include "mindspore/core/include/ir/func_graph_flag.h"
#include "mindspore/core/include/utils/ms_context.h"
#include "include/frontend/jit/ps/parse/py_data_convert.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "include/frontend/jit/ps/resource_interface.h"
#include "include/frontend/jit/ps/pipeline_interface.h"
#include "include/utils/python_adapter.h"
#include "include/utils/tensor_py.h"
#include "mindspore/ccsrc/pybind_api/backend/backend_api.h"

#ifdef ENABLE_DUMP_IR
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#endif

namespace mindspore {
namespace backend {
namespace py = pybind11;
class BackendFuncGraphMock;
using BackendFuncGraphMockPtr = std::shared_ptr<BackendFuncGraphMock>;
class BackendFuncGraphMock {
 public:
  BackendFuncGraphMock()
      : func_graph_{std::make_shared<FuncGraph>()}, resource_{std::make_shared<pipeline::Resource>()} {}
  ~BackendFuncGraphMock() {}
  std::string ToString() {
    std::ostringstream ofs;
    DumpIR(ofs, func_graph_, true);
    return ofs.str();
  }

  size_t AddWeightParameter(const py::object &obj) {
    if (!tensor::IsTensorPy(obj)) {
      MS_LOG(ERROR) << "Invalid Parameter:" << obj;
      return SIZE_MAX;
    }
    const auto &tensor = tensor::ConvertToTensor(obj);
    MS_EXCEPTION_IF_NULL(tensor);
    if (!tensor->is_parameter()) {
      MS_LOG(ERROR) << "Invalid Parameter:" << obj;
      return SIZE_MAX;
    }
    size_t index = nodes_.size();
    MS_EXCEPTION_IF_NULL(func_graph_);
    const auto &parameter = func_graph_->add_parameter();
    MS_EXCEPTION_IF_NULL(parameter);
    parameter->set_default_param(tensor);
    nodes_[index] = parameter;
    parameter->set_abstract(tensor->ToAbstract());
    func_graph_->AddNode(nodes_[index]);
    return index;
  }

  size_t AddParameter(const py::object &type_obj, const py::tuple &shape_obj) {
    MS_EXCEPTION_IF_NULL(func_graph_);
    size_t index = nodes_.size();
    nodes_[index] = func_graph_->add_parameter();
    MS_EXCEPTION_IF_NULL(nodes_[index]);
    if (!py::isinstance<mindspore::Type>(type_obj)) {
      MS_LOG(ERROR) << "Invalid type:" << type_obj;
      return SIZE_MAX;
    }
    TypePtr type = py::cast<mindspore::TypePtr>(type_obj);
    MS_EXCEPTION_IF_NULL(type);
    ShapeVector shape;
    for (size_t i = 0; i < shape_obj.size(); ++i) {
      if (!py::isinstance<py::int_>(shape_obj[i])) {
        MS_LOG(ERROR) << "Invalid shape" << shape_obj;
        return SIZE_MAX;
      }
      shape.emplace_back(py::cast<int64_t>(shape_obj[i]));
    }
    func_graph_->AddNode(nodes_[index]);
    nodes_[index]->set_abstract(std::make_shared<abstract::AbstractTensor>(type, shape));
    return index;
  }

  size_t AddValueNode(const py::object &obj) {
    MS_EXCEPTION_IF_NULL(func_graph_);
    if (py::isinstance<BackendFuncGraphMock>(obj)) {
      auto backend_func_graph = py::cast<BackendFuncGraphMockPtr>(obj);
      MS_EXCEPTION_IF_NULL(backend_func_graph);
      auto func_graph = backend_func_graph->func_graph_;
      const auto &value_node = NewValueNode(func_graph);
      size_t index = nodes_.size();
      nodes_[index] = value_node;
      func_graph_->AddNode(nodes_[index]);
      return index;
    }
    ValuePtr value = parse::data_converter::PyDataToValue(obj);
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<PrimitivePy>()) {
      auto prim_value = value->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(prim_value);
      value = std::make_shared<Primitive>(*prim_value);
      MS_EXCEPTION_IF_NULL(value);
    }
    const auto &abstract = value->ToAbstract();
    const auto &value_node = NewValueNode(value);
    MS_EXCEPTION_IF_NULL(value_node);
    value_node->set_abstract(abstract);
    size_t index = nodes_.size();
    nodes_[index] = value_node;
    func_graph_->AddNode(nodes_[index]);
    return index;
  }

  size_t AddCNode(const py::tuple &tuple) {
    if (tuple.size() == 0) {
      MS_LOG(ERROR) << "No input nodes.";
      return SIZE_MAX;
    }
    AnfNodePtrList cnode_inputs;
    if (py::isinstance<py::str>(tuple[0])) {
      cnode_inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(py::cast<std::string>(tuple[0]))));
    } else if (py::isinstance<py::int_>(tuple[0])) {
      size_t index = IntToSize(py::cast<int>(tuple[0]));
      if (nodes_.find(index) == nodes_.end()) {
        MS_LOG(ERROR) << "Failed to get node of index:" << index;
        return SIZE_MAX;
      }
      cnode_inputs.emplace_back(nodes_[index]);
    } else {
      MS_LOG(ERROR) << "Expect str phase, but got " << py::str(tuple[0]);
      return SIZE_MAX;
    }
    for (size_t i = 1; i < tuple.size(); ++i) {
      const auto &input = tuple[i];
      if (!py::isinstance<py::int_>(input)) {
        MS_LOG(ERROR) << "Expect int phase, but got " << py::str(input);
        return SIZE_MAX;
      }
      size_t index = IntToSize(py::cast<int>(input));
      if (nodes_.find(index) == nodes_.end()) {
        MS_LOG(ERROR) << "Failed to get node of index:" << index;
        return SIZE_MAX;
      }
      auto anf_input = nodes_[index];
      MS_EXCEPTION_IF_NULL(anf_input);
      cnode_inputs.emplace_back(anf_input);
    }
    size_t index = nodes_.size();
    MS_EXCEPTION_IF_NULL(func_graph_);
    nodes_[index] = func_graph_->NewCNode(cnode_inputs);
    func_graph_->AddNode(nodes_[index]);
    return index;
  }

  void AddReturn(const py::object &input) {
    if (!py::isinstance<py::int_>(input)) {
      MS_LOG(ERROR) << "Expect int phase, but got " << py::str(input);
      return;
    }
    size_t index = IntToSize(py::cast<int>(input));
    if (nodes_.find(index) == nodes_.end()) {
      MS_LOG(ERROR) << "Failed to get node of index:" << index;
      return;
    }
    auto anf_input = nodes_[index];
    MS_EXCEPTION_IF_NULL(anf_input);
    std::vector<AnfNodePtr> return_inputs{NewValueNode(std::make_shared<Primitive>("Return")), anf_input};
    MS_EXCEPTION_IF_NULL(func_graph_);
    auto return_node = func_graph_->NewCNode(return_inputs);
    func_graph_->AddNode(return_node);
    func_graph_->set_return(return_node);
  }

  void AddSubGraph(const py::object &obj) {
    if (!py::isinstance<BackendFuncGraphMock>(obj)) {
      MS_LOG(ERROR) << "Invalid sub graph:" << obj;
      return;
    }
    auto backend_func_graph = py::cast<BackendFuncGraphMockPtr>(obj);
    MS_EXCEPTION_IF_NULL(backend_func_graph);
    auto func_graph = backend_func_graph->func_graph_;
    MS_EXCEPTION_IF_NULL(resource_);
    auto manager = resource_->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(func_graph);
  }

  void SetCellReuse() {
    MS_EXCEPTION_IF_NULL(func_graph_);
    func_graph_->set_flag(FUNC_GRAPH_FLAG_CELL_REUSE, true);
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    context->SetCellReuseLevel(CellReuseLevel::kLazyInline);
  }

  void SetAbstract(const py::tuple &tuple) {
    constexpr size_t kInputSizeOfSetAbstractByTypeAndShape = 3;
    constexpr size_t kInputSizeOfSetAbstractByNode = 2;
    constexpr size_t kTypeIndex = 1;
    constexpr size_t kShapeIndex = 2;
    if (tuple.size() == kInputSizeOfSetAbstractByTypeAndShape) {
      if (!py::isinstance<py::int_>(tuple[0])) {
        MS_LOG(ERROR) << "Expect int phase, but got " << tuple;
        return;
      }
      size_t src_index = IntToSize(py::cast<int>(tuple[0]));
      if (nodes_.find(src_index) == nodes_.end() || nodes_[src_index] == nullptr) {
        MS_LOG(ERROR) << "Failed to get node of index:" << src_index;
        return;
      }
      if (!py::isinstance<mindspore::Type>(tuple[kTypeIndex])) {
        MS_LOG(ERROR) << "Invalid type:" << tuple;
        return;
      }
      TypePtr type = py::cast<mindspore::TypePtr>(tuple[kTypeIndex]);
      MS_EXCEPTION_IF_NULL(type);
      ShapeVector shape;
      if (!py::isinstance<py::tuple>(tuple[kShapeIndex])) {
        MS_LOG(ERROR) << "Invalid shape:" << tuple;
        return;
      }
      py::tuple shape_obj = py::cast<py::tuple>(tuple[kShapeIndex]);
      for (size_t i = 0; i < shape_obj.size(); ++i) {
        if (!py::isinstance<py::int_>(shape_obj[i])) {
          MS_LOG(ERROR) << "Invalid shape" << shape_obj;
          return;
        }
        shape.emplace_back(py::cast<int64_t>(shape_obj[i]));
      }
      nodes_[src_index]->set_abstract(std::make_shared<abstract::AbstractTensor>(type, shape));
      return;
    }
    if (tuple.size() != kInputSizeOfSetAbstractByNode) {
      MS_LOG(ERROR) << "Failed to set abstract for:" << tuple;
      return;
    }
    py::object src = tuple[0];
    py::object dst = tuple[1];
    if (!py::isinstance<py::int_>(src)) {
      MS_LOG(ERROR) << "Expect int phase, but got " << py::str(src);
      return;
    }
    size_t src_index = IntToSize(py::cast<int>(src));
    if (nodes_.find(src_index) == nodes_.end() || nodes_[src_index] == nullptr) {
      MS_LOG(ERROR) << "Failed to get node of index:" << src_index;
      return;
    }
    if (!py::isinstance<py::int_>(dst)) {
      MS_LOG(ERROR) << "Expect int phase, but got " << py::str(dst);
      return;
    }
    size_t dst_index = IntToSize(py::cast<int>(dst));
    if (nodes_.find(dst_index) == nodes_.end() || nodes_[dst_index] == nullptr) {
      MS_LOG(ERROR) << "Failed to get node of index:" << dst_index;
      return;
    }
    nodes_[dst_index]->set_abstract(nodes_[src_index]->abstract());
  }

  void SetTarget(const py::object &cnode_obj, const py::object &target_obj) {
    if (!py::isinstance<py::int_>(cnode_obj)) {
      MS_LOG(ERROR) << "Expect int phase, but got " << py::str(cnode_obj);
      return;
    }
    size_t node_index = IntToSize(py::cast<int>(cnode_obj));
    if (nodes_.find(node_index) == nodes_.end() || nodes_[node_index] == nullptr) {
      MS_LOG(ERROR) << "Failed to get node of index:" << node_index;
      return;
    }
    if (!py::isinstance<py::str>(target_obj)) {
      MS_LOG(ERROR) << "Expect int phase, but got " << py::str(cnode_obj);
      return;
    }
    auto target = py::cast<std::string>(target_obj);
    auto cnode = nodes_[node_index]->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_user_data("primitive_target", std::make_shared<std::string>(target));
  }

  void SetInput(const py::object &cnode_obj, const py::object &index_obj, const py::object &input_obj) {
    if (!py::isinstance<py::int_>(cnode_obj)) {
      MS_LOG(ERROR) << "Expect int phase, but got " << py::str(cnode_obj);
      return;
    }
    size_t cnode_index = IntToSize(py::cast<int>(cnode_obj));
    if (nodes_.find(cnode_index) == nodes_.end() || nodes_[cnode_index] == nullptr ||
        !nodes_[cnode_index]->isa<CNode>()) {
      MS_LOG(ERROR) << "Invalid cnode of index:" << cnode_index;
      return;
    }
    if (!py::isinstance<py::int_>(input_obj)) {
      MS_LOG(ERROR) << "Expect int phase, but got " << py::str(input_obj);
      return;
    }
    size_t input_node_index = IntToSize(py::cast<int>(input_obj));
    if (nodes_.find(input_node_index) == nodes_.end() || nodes_[input_node_index] == nullptr) {
      MS_LOG(ERROR) << "Failed to get node of index:" << input_node_index;
      return;
    }
    if (!py::isinstance<py::int_>(index_obj)) {
      MS_LOG(ERROR) << "Expect int phase, but got " << py::str(index_obj);
      return;
    }
    size_t input_index = IntToSize(py::cast<int>(index_obj));
    const auto &cnode = nodes_[cnode_index]->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_input(input_index, nodes_[input_node_index]);
  }

  void Infer() {
    MS_EXCEPTION_IF_NULL(resource_);
    MS_EXCEPTION_IF_NULL(func_graph_);
    auto manager = resource_->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(func_graph_);
    resource_->set_func_graph(func_graph_);
    resource_->set_pipeline_level(pipeline::kLevelJit);
    abstract::AbstractBasePtrList args_list;
    std::for_each(func_graph_->parameters().begin(), func_graph_->parameters().end(),
                  [&args_list](const AnfNodePtr &para) { args_list.emplace_back(para->abstract()); });
    func_graph_ = pipeline::Renormalize(resource_, func_graph_, args_list);
    MS_EXCEPTION_IF_NULL(func_graph_);
  }

  std::vector<abstract::AbstractBasePtr> FetchInputAbstracts(const CNodePtr &cnode) {
    MS_EXCEPTION_IF_NULL(cnode);
    std::vector<abstract::AbstractBasePtr> abstracts{};
    for (size_t i = 1; i < cnode->size(); ++i) {
      const auto &input = cnode->inputs()[i];
      MS_EXCEPTION_IF_NULL(input);
      const auto &abstract = input->abstract();
      if (abstract == nullptr) {
        MS_LOG_WITH_NODE(EXCEPTION, input) << "Invalid abstract for input:" << input->DebugString()
                                           << " for node:" << cnode->fullname_with_scope() << " input index:" << i;
      }
      MS_LOG(DEBUG) << "Add abstract:" << abstract->ToString() << " for input:" << input->DebugString();
      abstracts.emplace_back(abstract);
    }
    return abstracts;
  }

  void NativeInfer() {
    MS_EXCEPTION_IF_NULL(resource_);
    MS_EXCEPTION_IF_NULL(func_graph_);
    auto manager = resource_->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(func_graph_);
    resource_->set_func_graph(func_graph_);
    resource_->set_pipeline_level(pipeline::kLevelJit);
    AnfNodePtrList nodes = TopoSort(func_graph_->get_return());
    for (const auto &node : nodes) {
      if (node == nullptr || (!node->isa<CNode>())) {
        continue;
      }
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->inputs().empty() || !IsValueNode<Primitive>(cnode->input(0))) {
        continue;
      }
      cnode->set_abstract(nullptr);
      MS_LOG(DEBUG) << "Infer abstract for node:" << node->fullname_with_scope();
      // Fetch input abstracts.
      std::vector<abstract::AbstractBasePtr> abstracts = FetchInputAbstracts(cnode);

      // Fetch infer function.
      const auto &primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
      MS_EXCEPTION_IF_NULL(primitive);
      auto abstract_opt = abstract::TryInferAbstract(primitive, abstracts);
      if (!abstract_opt.has_value()) {
        MS_LOG_WITH_NODE(EXCEPTION, cnode)
          << "Failed to infer for primitive:" << primitive->ToString() << " in node:" << cnode->fullname_with_scope();
      }
      auto abstract = abstract_opt.value();
      MS_LOG(INFO) << "Set abstract:" << abstract->ToString() << " for node:" << cnode->DebugString();
      cnode->set_abstract(abstract);
    }
  }

  void SkipInfer() {
    MS_EXCEPTION_IF_NULL(resource_);
    MS_EXCEPTION_IF_NULL(func_graph_);
    auto manager = resource_->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(func_graph_);
    resource_->set_func_graph(func_graph_);
    resource_->set_pipeline_level(pipeline::kLevelJit);
  }

  void SetJitConfig(const py::object &obj) {
    if (!py::isinstance<py::dict>(obj)) {
      MS_LOG(WARNING) << "Invalid config:" << obj;
      return;
    }
    std::map<std::string, std::string> jit_config;
    auto dick_obj = py::cast<py::dict>(obj);
    for (const auto &pair : dick_obj) {
      if (!py::isinstance<py::str>(pair.first) || !py::isinstance<py::str>(pair.second)) {
        MS_LOG(WARNING) << "Invalid config:" << obj;
        return;
      }
      jit_config[py::cast<std::string>(py::cast<py::str>(pair.first))] =
        py::cast<std::string>(py::cast<py::str>(pair.second));
    }
    PhaseManager::GetInstance().set_jit_config(jit_config);
  }

  void Compile() {
    MS_EXCEPTION_IF_NULL(resource_);
    MS_EXCEPTION_IF_NULL(func_graph_);
    pipeline::TaskEmitAction(resource_);
    pipeline::ExecuteAction(resource_);
  }

  py::object Run(const py::tuple &args) {
    MS_EXCEPTION_IF_NULL(resource_);
    MS_EXCEPTION_IF_NULL(func_graph_);
    if (!py::isinstance<py::tuple>(args)) {
      MS_LOG(EXCEPTION) << "Invalid input args:" << args;
    }
    ValuePtr inputs = nullptr;
    parse::ConvertData(args, &inputs);
    MS_EXCEPTION_IF_NULL(inputs);
    if (inputs == nullptr || !inputs->isa<ValueTuple>()) {
      MS_LOG(EXCEPTION) << "Invalid inputs:" << (inputs == nullptr ? "null" : inputs->ToString())
                        << " for args:" << args;
    }
    VectorRef input_list;
    const auto &tuple_inputs = inputs->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_inputs);
    for_each(tuple_inputs->value().begin(), tuple_inputs->value().end(),
             [&input_list](const ValuePtr &input) { input_list.emplace_back(input); });
    using RunPtr = std::shared_ptr<std::function<BaseRef(const VectorRef &)>>;
    auto run = resource_->GetResult("output").cast<RunPtr>();
    MS_EXCEPTION_IF_NULL(run);
    BaseRef output = (*run)(input_list);
    MS_EXCEPTION_IF_NULL(func_graph_->output());
    return pipeline::BaseRefToPyDataWithUserData(output, func_graph_->output()->abstract());
  }

  FuncGraphPtr func_graph_;
  std::unordered_map<size_t, AnfNodePtr> nodes_;
  pipeline::ResourcePtr resource_;
};

void RegBackendGraphMock(py::module *m) {
  (void)py::class_<BackendFuncGraphMock, BackendFuncGraphMockPtr>(*m, "BackendGraphMock_")
    .def(py::init([]() { return std::make_shared<BackendFuncGraphMock>(); }))
    .def("__str__", &BackendFuncGraphMock::ToString, "Executor ToString function.")
    .def("add_parameter_", &BackendFuncGraphMock::AddParameter, "Executor AddParameter function.")
    .def("add_weight_parameter_", &BackendFuncGraphMock::AddWeightParameter, "Executor run function.")
    .def("add_valuenode_", &BackendFuncGraphMock::AddValueNode, "Executor AddValueNode function.")
    .def("add_cnode_", &BackendFuncGraphMock::AddCNode, "Executor AddCNode function.")
    .def("add_return_", &BackendFuncGraphMock::AddReturn, "Executor AddReturn function.")
    .def("add_subgraph_", &BackendFuncGraphMock::AddSubGraph, "Executor AddSubGraph function.")
    .def("set_abstract_", &BackendFuncGraphMock::SetAbstract, "Executor SetAbstract function.")
    .def("set_cell_reuse_", &BackendFuncGraphMock::SetCellReuse, "Executor SetAbstract function.")
    .def("set_input_", &BackendFuncGraphMock::SetInput, "Executor SetAbstract function.")
    .def("set_target_", &BackendFuncGraphMock::SetTarget, "Executor SetAbstract function.")
    .def("set_jit_config_", &BackendFuncGraphMock::SetJitConfig, "Executor SetAbstract function.")
    .def("infer_", &BackendFuncGraphMock::Infer, "Executor Infer function.")
    .def("native_infer_", &BackendFuncGraphMock::NativeInfer, "Executor Infer function.")
    .def("skip_infer_", &BackendFuncGraphMock::SkipInfer, "Executor Infer function.")
    .def("compile_", &BackendFuncGraphMock::Compile, "Executor Compile function.")
    .def("__call__", &BackendFuncGraphMock::Run, "Executor Run function.");
}
}  // namespace backend
}  // namespace mindspore
