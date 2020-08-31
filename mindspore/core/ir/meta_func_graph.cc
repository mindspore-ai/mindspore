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
#include "base/core_ops.h"
#include "utils/ms_context.h"
#include "abstract/abstract_function.h"

// namespace to support intermediate representation definition
namespace mindspore {
abstract::AbstractBasePtr MetaFuncGraph::ToAbstract() {
  return std::make_shared<abstract::MetaFuncGraphAbstractClosure>(shared_from_base<MetaFuncGraph>());
}

FuncGraphPtr MetaFuncGraph::GenerateStubFunc(const TypePtrList &types) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_sparse = context->get_param<bool>(MS_CTX_ENABLE_SPARSE);
  if (!enable_sparse) {
    return nullptr;
  }

  std::vector<AnfNodePtr> parameters;
  ParameterPtr undetermined_param = nullptr;
  auto stub = std::make_shared<FuncGraph>();
  for (size_t i = 0; i < types.size(); ++i) {
    auto param = stub->add_parameter();
    parameters.push_back(param);
    if (types[i]->type_id() == kObjectTypeUndeterminedType) {
      undetermined_param = param;
    }
  }
  if (undetermined_param != nullptr) {
    std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < types.size(); ++i) {
      if (types[i]->type_id() == kObjectTypeFunction) {
        std::vector<AnfNodePtr> call_prim{parameters[i], undetermined_param};
        inputs.push_back(stub->NewCNode(call_prim));
      } else {
        inputs.push_back(parameters[i]);
      }
    }
    auto stub_output = stub->NewCNode(inputs);
    stub->set_output(stub_output);
    stub->set_stub(true);
    return stub;
  }
  return nullptr;
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
