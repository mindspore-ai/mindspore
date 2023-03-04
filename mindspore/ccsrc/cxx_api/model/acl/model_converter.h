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

#ifndef MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_MODEL_CONVERTER_H
#define MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_MODEL_CONVERTER_H
#include <string>
#include <map>
#include <memory>
#include "include/api/types.h"
#include "include/api/status.h"
#include "mindspore/core/ir/func_graph.h"
#include "include/transform/graph_ir/types.h"
#include "external/ge/ge_ir_build.h"
#include "cxx_api/model/acl/acl_model_options.h"

namespace mindspore {
class MS_API ModelConverter {
 public:
  ModelConverter() : options_() {}
  ~ModelConverter() = default;

  Buffer LoadMindIR(const FuncGraphPtr &func_graph);
  Buffer LoadMindIRSingle(const FuncGraphPtr &func_graph);

  void set_options(const std::weak_ptr<AclModelOptions> &options) { options_ = options; }

  Status SaveModel(const ge::ModelBufferData &model) const;

 private:
  transform::DfGraphPtr ConvertFuncGraphToAIR(const FuncGraphPtr &anf_graph) const;
  Buffer BuildAirModel(const transform::DfGraphPtr &graph, const std::map<std::string, std::string> &init_options,
                       const std::map<std::string, std::string> &build_options) const;
  Buffer LoadAscendIRInner(const Buffer &model_data);

  std::weak_ptr<AclModelOptions> options_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_MODEL_CONVERTER_H
