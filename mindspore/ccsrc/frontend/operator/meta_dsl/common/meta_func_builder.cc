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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_func_builder.h"
#include <algorithm>
#include <utility>
#include "ir/anf.h"
#include "ops/op_def.h"
#include "include/utils/ir_dump/anf_ir_dump_interface.h"
#include "frontend/operator/cc_implementations.h"
#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"
#include "ir/func_graph_flag.h"

namespace mindspore::prim {
void MetaFuncBuilder::BeginFunc() {
  MS_LOG(DEBUG) << "Begin function for " << name_;
  graph_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  MS_EXCEPTION_IF_NULL(graph_->debug_info());
  graph_->debug_info()->set_name(name_);
}

FuncGraphPtr MetaFuncBuilder::EndFunc() const {
  MS_LOG(DEBUG) << "End function for " << name_ << ", graph: " << graph_->ToString();
  return graph_;
}

AnfNodePtr MetaFuncBuilder::AddParameter(const std::string &name) {
  MS_LOG(DEBUG) << "Add parameter '" << name << "' to " << name_;
  auto param = graph_->add_parameter();
  param->set_name(name);
  return param;
}

void MetaFuncBuilder::SetOutput(const AnfNodePtr &output) {
  MS_LOG(DEBUG) << "Add output '" << output->DebugString() << "' to " << name_;
  graph_->set_output(output);
}

AnfNodePtr MetaFuncBuilder::CreateNode(const AnfNodePtrList &nodes) { return graph_->NewCNodeInOrder(nodes); }
}  // namespace mindspore::prim
