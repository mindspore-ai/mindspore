/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_EXPANDER_FALLBACK_FALLBACK_IRBUILDER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_EXPANDER_FALLBACK_FALLBACK_IRBUILDER_H_
#include <memory>
#include <string>
#include <unordered_set>
#include <functional>
#include <vector>

#include "include/common/expander/core/node.h"
#include "include/common/expander/core/emitter.h"

namespace mindspore {
namespace expander {
using SelectKernelFunc = std::function<bool(const CNodePtr &)>;
class FallbackIRBuilder;
using IRBuilderFunc = std::function<NodePtrList(FallbackIRBuilder *)>;
struct IRBuilderHandle {
  IRBuilderFunc func;
};

class FallbackIRBuilder : public IrEmitter {
 public:
  FallbackIRBuilder(const std::string &name, const FuncGraphPtr &fg, const SelectKernelFunc &func);
  ~FallbackIRBuilder() override = default;

  AnfNodePtr Run(const CNodePtr &cnode, const IRBuilderHandle &handle);
  ValuePtr GetAttr(const std::string &attr) const;
  TypePtr GetDtype(const NodePtr &node) const { return node->dtype(); }
  template <typename S>
  S GetAttr(const std::string &attr) const {
    return GetValue<S>(GetAttr(attr));
  }
  const mindspore::HashMap<std::string, ValuePtr> &GetAttrs() const { return *attrs_ptr_; }
  NodePtr GetInput(size_t i) const {
    if (i >= inputs_.size()) {
      MS_LOG(EXCEPTION) << "For " << name_ << ", the index " << i << " is out of range of inputs size "
                        << inputs_.size();
    }
    return inputs_[i];
  }
  const NodePtrList &GetInputs() const { return inputs_; }
  int64_t GetSize(const NodePtr &node) const;
  ShapeVector GetShape(const NodePtr &node) const { return node->shape(); }
  NodePtr DynSize(const NodePtr &node, const TypePtr &type);
  NodePtr DynSize(const NodePtr &node);
  NodePtr SequenceToTensor(const NodePtr &node, const TypePtr &dtype = kInt64);
  std::vector<int64_t> GetIntList(const ValuePtr &value);
  std::vector<int64_t> GetIntList(const NodePtr &node);

 protected:
  std::string name_;
  NodePtrList inputs_;
  const mindspore::HashMap<std::string, ValuePtr> *attrs_ptr_{nullptr};
  bool success_{true};
};

class IRBuilderFactory {
 public:
  static IRBuilderFactory &Instance() {
    static IRBuilderFactory instance{};
    return instance;
  }

  const IRBuilderHandle *GetBuilder(const std::string &name) const {
    auto iter = registry_.find(name);
    return (iter == registry_.end()) ? nullptr : &(iter->second);
  }

  class RegHelper {
   public:
    explicit RegHelper(const std::string &name) : name_(name) {}
    ~RegHelper() = default;
    const RegHelper &SetBody(const IRBuilderFunc &func) const {
      IRBuilderFactory::Instance().registry_[name_].func = func;
      return *this;
    }

   private:
    std::string name_;
  };

 private:
  HashMap<std::string, IRBuilderHandle> registry_;
};

#define EXPANDER_JOIN(x, y) x##y
#define EXPANDER_UNIQUE_NAME(prefix, cnt) EXPANDER_JOIN(prefix, cnt)
#define REG_FALLBACK_BUILDER(name) \
  static const IRBuilderFactory::RegHelper EXPANDER_UNIQUE_NAME(g_fbib, __COUNTER__) = IRBuilderFactory::RegHelper(name)
#define BODYFUNC(v) [](FallbackIRBuilder * v) -> NodePtrList
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_EXPANDER_FALLBACK_FALLBACK_IRBUILDER_H_
