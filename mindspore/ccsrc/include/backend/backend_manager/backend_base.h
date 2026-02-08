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
#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_BACKEND_MANAGER_BACKEND_BASE_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_BACKEND_MANAGER_BACKEND_BASE_H_

#include <memory>
#include <string>
#include <map>
#include <vector>
#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "mindspore/core/include/base/base_ref.h"
#include "backend/backend_manager/visible.h"
#include "include/backend/backend_manager/backend_jit_config.h"
#include "ir/tensor.h"

namespace mindspore {
namespace backend {
using BackendGraphId = uint32_t;

// The return value enum of BackendBase::Run.
enum RunningStatus {
  kRunningSuccess = 0,
  kRunningFailure,
};

enum IRFormat {
  kAir = 0,
};

struct GraphFragment;
namespace py = pybind11;
using GraphFragmentPtr = std::shared_ptr<GraphFragment>;
using FragmentRunFunc = std::function<RunningStatus(GraphFragment *frag, const VectorRef &inputs, VectorRef *outputs)>;
using PyToValueConvertFunc = std::function<bool(const py::object &obj, ValuePtr *value)>;
using ValueToPyConvertFunc = std::function<py::object(const BaseRef &value)>;

struct GraphFragment {
  GraphFragment() {}
  GraphFragment(size_t id, bool is_graph, const std::string &key,
                const std::vector<std::pair<int, std::string>> &args_list, FragmentRunFunc runner) {
    id_ = id;
    is_graph_ = is_graph;
    key_ = key;
    args_list_ = args_list;
    runner_ = runner;
  }
  py::object id() { return py::int_(id_); }
  py::object is_graph() { return py::bool_(is_graph_); }
  py::object py_key() { return py::str(key_); }
  py::object args_list() {
    py::list args = py::list();
    for (const auto &pair : args_list_) {
      auto py_pair = py::tuple(2);
      py_pair[0] = pair.first;
      if (pair.first >= 0) {
        py_pair[1] = std::atoi(pair.second.c_str());
      } else {
        py_pair[1] = pair.second;
      }
      args.append(py_pair);
    }

    return args;
  }
  explicit GraphFragment(const GraphFragmentPtr &frag) {
    MS_EXCEPTION_IF_NULL(frag);
    id_ = frag->id_;
    is_graph_ = frag->is_graph_;
    key_ = frag->key_;
    args_list_ = frag->args_list_;
    runner_ = frag->runner_;
    py_to_value_converter_ = frag->py_to_value_converter_;
    value_to_py_converter_ = frag->value_to_py_converter_;
  }
  py::object Run(const py::tuple &args) {
    if (runner_ == nullptr || py_to_value_converter_ == nullptr || value_to_py_converter_ == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid run handler for graph fragment:" << id_;
    }
    if (!py::isinstance<py::tuple>(args)) {
      MS_LOG(EXCEPTION) << "Invalid input args:" << args << " for fragment:" << id_;
    }
    ValuePtr inputs = nullptr;
    py_to_value_converter_(args, &inputs);
    MS_EXCEPTION_IF_NULL(inputs);
    if (inputs == nullptr || !inputs->isa<ValueTuple>()) {
      MS_LOG(EXCEPTION) << "Invalid inputs:" << (inputs == nullptr ? "null" : inputs->ToString())
                        << " for args:" << args << " for fragment:" << id_;
    }
    VectorRef input_list;
    const auto &tuple_inputs = inputs->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_inputs);
    for_each(tuple_inputs->value().begin(), tuple_inputs->value().end(),
             [&input_list](const ValuePtr &input) { input_list.emplace_back(input); });
    VectorRef outputs;
    runner_(this, input_list, &outputs);
    return value_to_py_converter_(outputs[0]);
  }
  std::string ToString() {
    std::stringstream buf;
    buf << "fragment id:" << id_ << " is graph:" << is_graph_ << " key:" << key_ << " args:" << args_list();
    return buf.str();
  }
  size_t id_{SIZE_MAX};
  bool is_graph_{true};
  std::string key_;
  std::vector<std::pair<int, std::string>> args_list_;
  FragmentRunFunc runner_{nullptr};
  BackendGraphId graph_id_{UINT32_MAX};
  PyToValueConvertFunc py_to_value_converter_{nullptr};
  ValueToPyConvertFunc value_to_py_converter_{nullptr};
};

// The base class of all supported backend.
class BACKEND_MANAGER_EXPORT BackendBase {
 public:
  // The backend graph Build interface, the return value is the built graph id.
  virtual std::vector<GraphFragmentPtr> Split(const FuncGraphPtr &func_graph) {
    MS_LOG(EXCEPTION) << "Not support graph split.";
  }
  // The backend graph Build interface, the return value is the built graph id.
  virtual BackendGraphId Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) = 0;

  // The backend graph Run interface by the graph_id which are generated through the graph Build interface above.
  virtual RunningStatus Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) = 0;

  // convert mindir to ir_format
  virtual void ConvertIR(const FuncGraphPtr &anf_graph,
                         const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                         IRFormat ir_format) {
    return;
  }

  // export graph to ir_format. If is_save_to_file=True, save as file; if False, return as string
  virtual std::string ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file,
                               IRFormat ir_format) {
    return "";
  }

  // clear the resource, init is in constructor function
  virtual void Clear() {}

  // clear the compiler info for graph
  virtual void ClearGraph(BackendGraphId backend_graph_id) {}
};

using BackendBasePtr = std::shared_ptr<BackendBase>;
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_BACKEND_MANAGER_BACKEND_BASE_H_
