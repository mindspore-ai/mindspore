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
#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PY_PASS_MANAGER_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PY_PASS_MANAGER_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "pybind_api/ir/primitive_py.h"
#include "ir/graph_utils.h"
#include "utils/ms_utils.h"

#include "pipeline/jit/resource.h"
#include "frontend/optimizer/pattern.h"
#include "frontend/optimizer/py_pass.h"
#include "frontend/optimizer/pass_group.h"

namespace mindspore {
namespace opt {
namespace python_pass {
class PyPassManager;
using PyPassManagerPtr = std::shared_ptr<PyPassManager>;

enum Phase { PREAD, OPT };

class PyPassManager {
 protected:
  PyPassManager();
  static PyPassManagerPtr global_instance;

 public:
  // Singletons should not be cloneable and assignable
  PyPassManager(const PyPassManager &other) = delete;
  void operator=(const PyPassManager &) = delete;
  // Access the only global instance
  static PyPassManagerPtr GetInstance();
  virtual ~PyPassManager() = default;
  void Register(const std::string &pass_name, const PatternPtr &pattern, const PatternPtr &target, bool requires_grad,
                bool run_only_once);
  void Unregister(const std::string &pass_name);
  void GenNewParameter(const PatternPtr &parameter);
  PassGroupPtr GetPassGroup(Phase phase);
  MatchResultPtr GetMatchResult() { return res_; }
  void SetRenorm(bool should_renorm) { should_renorm_ = should_renorm; }
  bool ShouldRenorm() { return should_renorm_; }
  void SetReOpt(bool should_reopt) { should_reopt_ = should_reopt; }
  bool ShouldReOpt() { return should_reopt_; }
  void SetResource(pipeline::ResourcePtr resource) { resource_ = resource; }
  pipeline::ResourcePtr GetResource() { return resource_; }
  void ClearRes();
  void ClearPipelineRes() {
    resource_ = nullptr;
    Pattern::reset_gid();
  }

 private:
  bool should_renorm_ = true;
  bool should_reopt_ = true;
  MatchResultPtr res_;
  pipeline::ResourcePtr resource_;
  static std::unordered_map<Phase, PassGroupPtr> phase_to_group_;
};
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PY_PASS_MANAGER_H_
