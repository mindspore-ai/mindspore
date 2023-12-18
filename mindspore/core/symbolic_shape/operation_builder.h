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
#ifndef MINDSPORE_CORE_SYMBOLIC_SHAPE_OPERATION_BUILDER_H_
#define MINDSPORE_CORE_SYMBOLIC_SHAPE_OPERATION_BUILDER_H_
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "mindspore/core/ir/primitive.h"
#include "mindspore/core/symbolic_shape/symbol.h"
#include "mindspore/core/symbolic_shape/operation.h"

namespace mindspore {
namespace symshape {
class OperationBuilder;
enum class DependOn : int { kShape, kValue, kNone };
using InferFunc = std::function<SymbolPtr(OperationBuilder *)>;
using DependFunc = std::function<std::vector<DependOn>(const PrimitivePtr &)>;
struct MS_CORE_API OperationBuilderInfo {
  InferFunc build_shape_func{nullptr};
  InferFunc build_value_func{nullptr};
  DependFunc shape_depend_func{nullptr};
  DependFunc value_depend_func{nullptr};
  std::vector<DependOn> shape_depend_list;
  std::vector<DependOn> value_depend_list;
  std::vector<DependOn> GetDepends(const PrimitivePtr &prim, bool build_value) const {
    return build_value ? (value_depend_func != nullptr ? value_depend_func(prim) : value_depend_list)
                       : (shape_depend_func != nullptr ? shape_depend_func(prim) : shape_depend_list);
  }
};

class MS_CORE_API OperationBuilder {
 public:
  OperationBuilder(OperationEmitter *emitter, const OperationBuilderInfo &info)
      : emitter_(emitter), symbol_builder_info_(info) {}
  ~OperationBuilder() = default;
  SymbolPtr BuildShape(const PrimitivePtr &prim, const AbstractBasePtrList &input_args, const AbstractBasePtr &out);
  SymbolPtr BuildValue(const PrimitivePtr &prim, const AbstractBasePtrList &input_args, const AbstractBasePtr &out);

  SymbolPtr Emit(const OpPtr &op) const { return emitter_->Emit(op); }
  SymbolPtr GetShape(const AbstractBasePtr &abs) const;
  SymbolPtr GetValue(const AbstractBasePtr &abs) const;
  const AbstractBasePtr &GetInput(size_t i) const { return input_args_->at(i); }
  SymbolPtr GetInputShape(size_t i) const { return GetShape(GetInput(i)); }
  SymbolPtr GetInputValue(size_t i) const { return GetValue(GetInput(i)); }
  SymbolPtr GetAttr(const std::string &attr_name) const;
  // todo, remove this after converting all attrs to input.
  SymbolPtr GetInputOrAttr(size_t index, const std::string &attr_name) const;

  bool is_building_shape() const { return is_building_shape_; }
  const PrimitivePtr &prim() const { return prim_; }
  size_t input_num() const { return input_args_->size(); }
  const AbstractBasePtr &out_abstract() const { return out_; }
  const OperationBuilderInfo &symbol_builder_info() const { return symbol_builder_info_; }

 protected:
  SymbolPtr TransparentInput(const PrimitivePtr &prim, bool build_value) const;

  OperationEmitter *emitter_;
  const OperationBuilderInfo &symbol_builder_info_;
  bool is_building_shape_{false};
  PrimitivePtr prim_;
  const AbstractBasePtrList *input_args_;
  AbstractBasePtr out_;
};
using OperationBuilderPtr = std::unique_ptr<OperationBuilder>;

class MS_CORE_API OperationBuilderInfoRegistry {
 public:
  static const OperationBuilderInfo *GetBuildInfo(const std::string &name);
  static OperationBuilderPtr GetBuilder(const std::string &name, OperationEmitter *e);

  static OperationBuilderInfoRegistry &Instance() {
    static OperationBuilderInfoRegistry instance{};
    return instance;
  }

  class RegHelper {
   public:
    explicit RegHelper(const std::string &name) : builder_(OperationBuilderInfoRegistry::Instance().NewBuilder(name)) {}
    RegHelper &SetShapeDepend(const std::initializer_list<DependOn> &depends) {
      builder_->shape_depend_list = depends;
      return *this;
    }
    RegHelper &SetShapeDepend(const DependFunc &func) {
      builder_->shape_depend_func = func;
      return *this;
    }
    RegHelper &SetShapeFunc(const InferFunc &func) {
      builder_->build_shape_func = func;
      return *this;
    }

    RegHelper &SetValueDepend(const std::initializer_list<DependOn> &depends) {
      builder_->value_depend_list = depends;
      return *this;
    }
    RegHelper &SetValueDepend(const DependFunc &func) {
      builder_->value_depend_func = func;
      return *this;
    }
    RegHelper &SetValueFunc(const InferFunc &func) {
      builder_->build_value_func = func;
      return *this;
    }
    OperationBuilderInfo *builder_;
  };  // class RegHelper

 private:
  OperationBuilderInfo *NewBuilder(const std::string &name) { return &builders_[name]; }
  std::unordered_map<std::string, OperationBuilderInfo> builders_;
};

#define JOIN(x, y) x##y
#define UNIQUE_NAME(prefix, cnt) JOIN(prefix, cnt)
#define REG_SYMBOL_OP_BUILDER(name) \
  const auto UNIQUE_NAME(g_ob_, __COUNTER__) = OperationBuilderInfoRegistry::RegHelper(name)
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_SYMBOLIC_SHAPE_OPERATION_BUILDER_H_
