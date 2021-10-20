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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ACL_ACL_PASS_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_ACL_ACL_PASS_H

#include <string>
#include <vector>
#include <memory>
#include "backend/optimizer/common/pass.h"
#include "include/errorcode.h"
#include "include/api/types.h"
#include "include/registry/converter_context.h"
#include "cxx_api/model/acl/acl_model_options.h"

namespace mindspore {
namespace opt {
using mindspore::converter::FmkType;
using mindspore::lite::STATUS;

class AclPass : public Pass {
 public:
  explicit AclPass(FmkType fmk_type) : Pass("Acl"), fmk_type_(fmk_type) {}
  ~AclPass() override = default;

  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  STATUS PreProcGraph(const FuncGraphPtr &func_graph);
  STATUS PostProcGraph(const FuncGraphPtr &func_graph);
  STATUS DeparseGraph(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
  STATUS RunPrimitiveMapper(const FuncGraphPtr &func_graph);
  STATUS BuildGraph(const FuncGraphPtr &func_graph);
  STATUS ConvertGraphToOm(const FuncGraphPtr &func_graph, Buffer *om_data);
  ParameterPtr CreateOmParameter(const FuncGraphPtr &func_graph, const Buffer &om);
  CNodePtr CreateCustomNode(const FuncGraphPtr &func_graph);
  STATUS SetCustomOutputs(const FuncGraphPtr &func_graph, const CNodePtr &custom_node);
  STATUS SetMultiOutputs(const CNodePtr &new_cnode, TypeId data_type);
  STATUS ModifyGraphByCustomNode(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                 const CNodePtr &custom_node);
  void SetAclModelOptions(const FuncGraphPtr &func_graph);
  STATUS GetFuncGraphOutputInfo(const FuncGraphPtr &func_graph);
  STATUS TraceOutput(const AnfNodePtr &node);

  FmkType fmk_type_;
  ParameterPtr om_parameter_ = nullptr;
  CNodePtr custom_node_ = nullptr;
  std::shared_ptr<AclModelOptions> options_;
  AnfNodePtrList graph_outputs_;
  std::vector<std::string> graph_output_names_;
  std::vector<std::vector<int64_t>> graph_output_dims_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ACL_ACL_PASS_H
