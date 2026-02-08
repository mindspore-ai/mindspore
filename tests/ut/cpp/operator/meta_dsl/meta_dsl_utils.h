/*
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

#ifndef MINDSPORE_CCSRC_TEST_UT_CPP_OPERATOR_META_DSL_META_DSL_UTILS_H_
#define MINDSPORE_CCSRC_TEST_UT_CPP_OPERATOR_META_DSL_META_DSL_UTILS_H_

#include "ir/manager.h"
#include "ir/graph_utils.h"
#include "ir/core_ops_primitive.h"
#include "common/common_test.h"
#include "frontend/jit/ps/action.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "frontend/optimizer/ad/grad.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "primitive/framework_ops.h"
#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"

namespace mindspore::prim {
MetaImplPtr CreateMetaImpl(const std::string &name) { return RegMetaImplFactory::GetInstance().CreateMetaImpl(name); }

inline FuncGraphPtr NewFuncGraph(const ValuePtr &op, const AbstractBasePtrList &abs_list) {
  // Create FuncGraph.
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  std::vector<FuncGraphPtr> graphs{fg};
  auto func_graph_manager = std::make_shared<FuncGraphManager>(graphs);
  AnfNodePtrList inputs{NewValueNode(op)};
  for (size_t i = 0; i < abs_list.size(); ++i) {
    auto param = fg->add_parameter();
    (void)inputs.emplace_back(param);
  }
  CNodePtr cnode = fg->NewCNode(inputs);
  fg->set_output(cnode);
  // Renormalize.
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  return pipeline::Renormalize(resource, fg, abs_list);
}

inline AbstractBasePtr RunMetaImpl(const std::vector<AbstractBasePtr> &args, const PrimitivePtr &prim) {
  auto fg = NewFuncGraph(prim, args);
  MS_EXCEPTION_IF_NULL(fg);
  return fg->return_node()->abstract();
}
}  // namespace mindspore::prim

#endif  // MINDSPORE_CCSRC_TEST_UT_CPP_OPERATOR_META_DSL_META_DSL_UTILS_H_