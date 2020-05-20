/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "ir/meta_func_graph.h"
#include "pipeline/static_analysis/static_analysis.h"
#include "pipeline/static_analysis/abstract_function.h"

// namespace to support intermediate representation definition
namespace mindspore {
abstract::AbstractBasePtr MetaFuncGraph::MakeAbstractClosure(const AnfNodePtr &anf_node) {
  abstract::MetaFuncGraphAbstractClosurePtr meta_func_graph_fn;
  if (anf_node == nullptr) {
    meta_func_graph_fn = std::make_shared<abstract::MetaFuncGraphAbstractClosure>(shared_from_base<MetaFuncGraph>());
  } else {
    meta_func_graph_fn =
      std::make_shared<abstract::MetaFuncGraphAbstractClosure>(shared_from_base<MetaFuncGraph>(), anf_node->scope());
  }
  return meta_func_graph_fn;
}

FuncGraphPtr MetaFuncGraph::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_spec_list) {
  TypePtrList types;
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(types),
                       [](const AbstractBasePtr &arg) -> TypePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         return arg->BuildType();
                       });
  // filter unsafe characters in log print since name_ is from outside
  auto iter = cache_.find(types);
  if (iter == cache_.end()) {
    FuncGraphPtr fg = GenerateFromTypes(types);
    MS_EXCEPTION_IF_NULL(fg);
    MS_LOG(INFO) << "MetaFuncgraph: cache miss for types: " << mindspore::ToString(args_spec_list)
                 << ", g: " << fg->ToString();
    cache_[types] = fg;
    return fg;
  } else {
    MS_LOG(DEBUG) << "MetaFuncgraph: cache hit for types: " << mindspore::ToString(args_spec_list)
                  << ", g: " << iter->second->ToString();
    return iter->second;
  }
}
}  // namespace mindspore
