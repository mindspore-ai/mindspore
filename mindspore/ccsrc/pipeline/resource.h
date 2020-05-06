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

#ifndef MINDSPORE_CCSRC_PIPELINE_RESOURCE_H_
#define MINDSPORE_CCSRC_PIPELINE_RESOURCE_H_

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "utils/any.h"
#include "utils/profile.h"
#include "ir/manager.h"
#include "pipeline/static_analysis/prim.h"
#include "pipeline/static_analysis/static_analysis.h"
#include "./common.h"

namespace mindspore {
namespace pipeline {

namespace py = pybind11;

const char kBackend[] = "backend";
const char kStepParallelGraph[] = "step_parallel";
const char kOutput[] = "output";

class InferenceResource;

using MethodMap = std::unordered_map<int, std::unordered_map<std::string, Any>>;

MethodMap &GetMethodMap();

class ResourceBase {
 public:
  ResourceBase() { manager_ = MakeManager(); }

  virtual ~ResourceBase() = default;

  FuncGraphManagerPtr manager() { return manager_; }
  // set a manager defined outside which will not manage the graphs.
  void set_manager(const FuncGraphManagerPtr &manager) { manager_ = manager; }

  std::unordered_map<std::string, Any> &results() { return results_; }

  void SetResult(const std::string &key, const Any &value) { results_[key] = value; }

  Any GetResult(const std::string &key) {
    if (results_.count(key) == 0) {
      MS_LOG(EXCEPTION) << "this key is not in resource list:" << key;
    }
    return results_[key];
  }

  bool HasResult(const std::string &key) const { return results_.count(key) != 0; }

  std::unordered_map<std::string, Any> results_;

 protected:
  FuncGraphManagerPtr manager_;
};

using ResourceBasePtr = std::shared_ptr<pipeline::ResourceBase>;

class Resource : public ResourceBase {
 public:
  explicit Resource(const py::object &obj = py::none());

  ~Resource() override;

  abstract::AnalysisEnginePtr engine() { return engine_; }

  static bool IsTypeInMethodMap(const TypeId &type);

  static Any GetMethodPtr(const TypeId &type, const std::string &name);

  const py::object &input() const { return input_; }

  FuncGraphPtr func_graph() const { return func_graph_; }
  void set_func_graph(const FuncGraphPtr &func_graph) { func_graph_ = func_graph; }

  const abstract::AbstractBasePtrList &args_spec() const { return args_spec_; }
  void set_args_spec(const abstract::AbstractBasePtrList &args_spec) { args_spec_ = args_spec; }

  // Reclaim resource and clear the cache.
  // ExecutorPy::Compile() can be called multiple times, so cache
  // should be cleared.
  void Clean();

 private:
  abstract::AnalysisEnginePtr engine_;
  FuncGraphPtr func_graph_;
  abstract::AbstractBasePtrList args_spec_;
  py::object input_;
  bool is_cleaned_;
};

using ResourcePtr = std::shared_ptr<pipeline::Resource>;

}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_RESOURCE_H_
