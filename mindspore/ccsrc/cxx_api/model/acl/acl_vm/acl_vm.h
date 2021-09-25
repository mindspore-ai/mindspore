/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_CXX_API_ACL_VM_ACL_VM_H
#define MINDSPORE_CCSRC_CXX_API_ACL_VM_ACL_VM_H

#include <vector>
#include <memory>
#include <string>
#include "vm/transform.h"
#include "vm/backend.h"
#include "cxx_api/model/acl/acl_vm/ms_tensor_ref.h"

namespace mindspore {
class AclModelOptions;
class AclBackend : public compile::MsBackend {
 public:
  AclBackend(const std::string &name, const std::string &target, const std::shared_ptr<AclModelOptions> &options);
  ~AclBackend() override = default;

  VectorRef MsRunGraph(const GraphId &g, const VectorRef &args, const std::string &target) override;
  bool GetCond(const BaseRef &c, bool *value) override;
  bool GetIndex(const BaseRef &c, int64_t *value) override;
};

class AclCompileGraph : public compile::CompileGraph {
 public:
  explicit AclCompileGraph(const std::shared_ptr<compile::MsBackend> &backend,
                           const std::vector<PrimitivePtr> &cut_list);
  ~AclCompileGraph() override = default;

  int64_t Ref(const AnfNodePtr &node) override;
  void AddExternal(const compile::LinConvertResult &result) override;
  void AddInput(const AnfNodePtr &node) override;
  void AddPartial(const CNodePtr &node) override;
  int64_t AddCall(const FuncGraphPtr &graph, const CNodePtr &node) override;
  void PushParameters(const FuncGraphPtr &func_graph) override;

 private:
  void AddInst(const compile::Instruction &inst, const MSTensorRef &arg);
};

class AclCompileGraphs : public compile::CompileGraphs {
 public:
  explicit AclCompileGraphs(const std::shared_ptr<compile::MsBackend> &backend,
                            const std::vector<PrimitivePtr> &cut_list);
  ~AclCompileGraphs() override = default;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_ACL_VM_ACL_VM_H
