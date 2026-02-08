/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_IRBUILDER_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_IRBUILDER_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <functional>

#include "include/frontend/expander/bprop_interface.h"
#include "mindspore/ccsrc/include/utils/expander/node.h"
#include "mindspore/ccsrc/include/utils/expander/emitter.h"

namespace mindspore {
namespace expander {
namespace bprop {
class IrBuilder : public BpropBuilder {
 public:
  IrBuilder(const std::string &name, const FuncGraphPtr &func_graph, const ExpanderInferPtr &infer)
      : BpropBuilder(name, infer), func_graph_(func_graph) {}
  NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) override;
  NodePtr EmitValue(const ValuePtr &value) override;
  const FuncGraphPtr &func_graph() { return func_graph_; }

 protected:
  FuncGraphPtr func_graph_;
};

class BpropIRBuilderFactory {
 public:
  static BpropIRBuilderFactory &Instance() {
    static BpropIRBuilderFactory instance{};
    return instance;
  }

  const BpropHandle *GetBuilder(const std::string &name) const {
    auto iter = registry_.find(name);
    return (iter == registry_.end()) ? nullptr : &(iter->second);
  }

  class RegHelper {
   public:
    explicit RegHelper(const std::string &name) : name_(name) {}
    ~RegHelper() = default;
    const RegHelper &SetBody(const BpropBuilderFunc &func) const {
      BpropIRBuilderFactory::Instance().RegBuilder(name_, func);
      return *this;
    }
    // DEPRECATED. use FreeUselessValues/FreeUselessValues_IO/FreeUselessValues_I/FreeUselessValues_O instead.
    const RegHelper &SetUnusedInputs(const std::initializer_list<size_t> &unused_inputs) const {
      BpropIRBuilderFactory::Instance().RegUnusedInputs(name_, unused_inputs);
      return *this;
    }
    /// \brief Register a function to free unused values before bprop execution. (pynative mode)
    const RegHelper &FreeUselessValues(const FreeUselessValueFunc &func) const {
      BpropIRBuilderFactory::Instance().RegFreeUselessValues(name_, func);
      return *this;
    }
    /// \brief Set the unused input indices.
    /// \param inputs_idx unused index of inputs. if empty, ALL inputs will be free.
    const RegHelper &FreeUselessValues_I(const std::initializer_list<size_t> &inputs_idx = {}) const {
      BpropIRBuilderFactory::Instance().RegFreeUselessValues(
        name_,
        [index = std::vector<size_t>(inputs_idx)](const PynativeCallback &cb) { cb.FreeInputDeviceAddress(index); });
      return *this;
    }
    /// \brief Set the unused output indices.
    /// \param outputs_idx unused index of outputs. if empty, ALL outputs will be free.
    const RegHelper &FreeUselessValues_O(const std::initializer_list<size_t> &outputs_idx = {}) const {
      BpropIRBuilderFactory::Instance().RegFreeUselessValues(
        name_,
        [index = std::vector<size_t>(outputs_idx)](const PynativeCallback &cb) { cb.FreeOutputDeviceAddress(index); });
      return *this;
    }
    /// \brief Set the unused input and output indices.
    /// \param inputs_idx unused index of inputs. if empty, ALL inputs will be free.
    /// \param outputs_idx unused index of outputs. if empty, ALL outputs will be free.
    const RegHelper &FreeUselessValues_IO(const std::initializer_list<size_t> &inputs_idx,
                                          const std::initializer_list<size_t> &outputs_idx) const {
      BpropIRBuilderFactory::Instance().RegFreeUselessValues(
        name_, [inputs = std::vector<size_t>(inputs_idx), outputs = std::vector<size_t>(outputs_idx)](
                 const PynativeCallback &cb) { cb.FreeIODeviceAddress(inputs, outputs); });
      return *this;
    }
    /// \brief Add whether clone inplace input.
    /// \param clone func.
    const RegHelper &CloneInplaceInput(const CloneInplaceInputFunc &func) const {
      BpropIRBuilderFactory::Instance().RegCloneInplaceInput(name_, func);
      return *this;
    }
    /// \brief Add whether clone inplace input.
    /// \param is clone flag.
    const RegHelper &CloneInplaceInput() const {
      BpropIRBuilderFactory::Instance().RegCloneInplaceInput(name_,
                                                             [](const PynativeCallback &cb) -> bool { return true; });
      return *this;
    }

   private:
    std::string name_;
  };

 private:
  void RegBuilder(const std::string &name, const BpropBuilderFunc &func) { registry_[name].func = func; }
  void RegUnusedInputs(const std::string &name, const mindspore::HashSet<size_t> &unused) {
    registry_[name].unused_inputs = unused;
  }
  void RegFreeUselessValues(const std::string &name, const FreeUselessValueFunc &func) {
    registry_[name].free_useless_value_func = func;
  }
  void RegCloneInplaceInput(const std::string &name, const CloneInplaceInputFunc &func) {
    registry_[name].clone_inplace_input_func = func;
  }
  HashMap<std::string, BpropHandle> registry_;
};

#define BPROP_EXPANDER_JOIN(x, y) x##y
#define BPROP_EXPANDER_UNIQUE_NAME(prefix, cnt) BPROP_EXPANDER_JOIN(prefix, cnt)
#define REG_BPROP_BUILDER(name) \
  const auto BPROP_EXPANDER_UNIQUE_NAME(g_bprop, __COUNTER__) = BpropIRBuilderFactory::RegHelper(name)
#define BODYFUNC(v) [](BpropBuilder * (v)) -> NodePtrList

#ifdef _MSC_VER
#define REG_BPROP_BUILDERS_BEGIN(func_name)
#define REG_BPROP_BUILDERS_END
#else
#define REG_BPROP_BUILDERS_BEGIN(func_name)
#define REG_BPROP_BUILDERS_END
#endif
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_IRBUILDER_H_
