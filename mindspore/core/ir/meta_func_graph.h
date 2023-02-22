/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_META_FUNC_GRAPH_H_
#define MINDSPORE_CORE_IR_META_FUNC_GRAPH_H_

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <algorithm>

#include "ir/dtype.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/signature.h"
#include "abstract/abstract_value.h"

namespace mindspore {
// namespace to support intermediate representation definition
// Graph generator.
// Can be called with a pipeline's resources and a list of argument types to
// generate a graph corresponding to these types.
class MS_CORE_API MetaFuncGraph : public FuncGraphBase {
 public:
  explicit MetaFuncGraph(const std::string &name) : name_(name) {
    cache_.clear();
    debug_info_ = std::make_shared<DebugInfo>();
  }

  ~MetaFuncGraph() { subclass_destruct_flag_ = true; }

  MS_DECLARE_PARENT(MetaFuncGraph, FuncGraphBase);
  // Return normalized versions of the arguments.
  // By default, this returns args unchanged.
  virtual abstract::AbstractBasePtrList NormalizeArgs(const abstract::AbstractBasePtrList &args_spec_list) const {
    return args_spec_list;
  }
  abstract::AbstractBasePtr ToAbstract() override;
  const std::vector<Signature> &signatures() const { return signatures_; }
  void set_signatures(const std::vector<Signature> &signatures) { signatures_ = signatures; }
  // Generate a Graph for the given abstract arguments.
  virtual FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &args_spec_list);

  // Generate a Graph for this type signature.
  virtual FuncGraphPtr GenerateFromTypes(const TypePtrList &) {
    MS_LOG(EXCEPTION) << "Undefined the method of generating graph from types. func_name:" << name();
  }

  std::string name() { return name_; }
  std::string ToString() const override {
    std::ostringstream buffer;
    buffer << "MetaFuncGraph-";
    buffer << name_;
    buffer << "." << debug_info_->get_id();
    return buffer.str();
  }
  std::size_t hash() const override { return tid(); }

  virtual bool operator==(const MetaFuncGraph &other) const { return &other == this; }
  bool operator==(const Value &other) const override {
    if (other.isa<MetaFuncGraph>()) {
      return &other == this;
    } else {
      return false;
    }
  }

  void DoBreakLoop() override { cache_.clear(); }

 protected:
  template <typename Derived>
  std::shared_ptr<Derived> shared_from_base() {
    return std::static_pointer_cast<Derived>(shared_from_this());
  }
  FuncGraphPtr GenerateStubFunc(const TypePtrList &types) const;
  std::string name_;
  std::vector<Signature> signatures_;
  TypeListMap<FuncGraphPtr> cache_;

 private:
  DebugInfoPtr debug_info_{nullptr};
};

using MetaFuncGraphPtr = std::shared_ptr<MetaFuncGraph>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_META_FUNC_GRAPH_H_
