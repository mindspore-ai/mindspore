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

#ifndef MINDSPORE_CCSRC_IR_META_FUNC_GRAPH_H_
#define MINDSPORE_CCSRC_IR_META_FUNC_GRAPH_H_

#include <unordered_map>
#include <string>
#include <map>
#include <memory>
#include <vector>
#include <algorithm>

#include "pybind11/pybind11.h"

#include "ir/dtype.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "pipeline/static_analysis/abstract_value.h"

namespace py = pybind11;

namespace mindspore {
// namespace to support intermediate representation definition
// Graph generator.
// Can be called with a pipeline's resources and a list of argument types to
// generate a graph corresponding to these types.
class MetaFuncGraph : public FuncGraphBase {
 public:
  explicit MetaFuncGraph(const std::string& name) : name_(name) { cache_.clear(); }

  ~MetaFuncGraph() override = default;

  MS_DECLARE_PARENT(MetaFuncGraph, FuncGraphBase);
  abstract::AbstractBasePtr MakeAbstractClosure(const AnfNodePtr& anf_node);
  // Return normalized versions of the arguments.
  // By default, this returns args unchanged.
  virtual abstract::AbstractBasePtrList NormalizeArgs(const abstract::AbstractBasePtrList& args_spec_list) const {
    return args_spec_list;
  }

  const std::vector<Signature>& signatures() const { return signatures_; }
  void set_signatures(const std::vector<Signature>& signatures) { signatures_ = signatures; }
  // Generate a Graph for the given abstract arguments.
  virtual FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList& args_spec_list) {
    TypePtrList types;
    (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(types),
                         [](const AbstractBasePtr& arg) -> TypePtr {
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

  // Generate a Graph for this type signature.
  virtual FuncGraphPtr GenerateFromTypes(const TypePtrList&) {
    MS_LOG(EXCEPTION) << "Undefine the method of generating graph from types.";
  }

  std::string name() { return name_; }
  std::string ToString() const override { return name_; }
  std::size_t hash() const override { return tid(); }

  virtual bool operator==(const MetaFuncGraph& other) const { return &other == this; }
  bool operator==(const Value& other) const override {
    if (other.isa<MetaFuncGraph>()) {
      return &other == this;
    } else {
      return false;
    }
  }
  const bool parse_info_ = true;

 protected:
  template <typename Derived>
  std::shared_ptr<Derived> shared_from_base() {
    return std::static_pointer_cast<Derived>(shared_from_this());
  }
  std::string name_;
  std::vector<Signature> signatures_;
  std::unordered_map<TypePtrList, FuncGraphPtr, TypeListHasher, TypeListEqual> cache_;
};

using MetaFuncGraphPtr = std::shared_ptr<MetaFuncGraph>;
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_META_FUNC_GRAPH_H_
