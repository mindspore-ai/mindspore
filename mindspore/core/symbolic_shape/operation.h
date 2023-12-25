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
#ifndef MINDSPORE_CORE_SYMBOLIC_SHAPE_OPERATION_H_
#define MINDSPORE_CORE_SYMBOLIC_SHAPE_OPERATION_H_
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <map>

#include "mindspore/core/symbolic_shape/symbol.h"

namespace mindspore {
namespace symshape {
/// \brief Operation is the basic class of operators for symbol.
class MS_CORE_API Operation : public Base {
 public:
  explicit Operation(SymbolPtrList &&inputs) : inputs_(inputs) {}
  virtual ~Operation() = default;
  MS_DECLARE_PARENT(Operation, Base)

  bool Build();
  inline void Run() {
    MS_LOG(DEBUG) << "Running operation " << ToString();
    MS_EXCEPTION_IF_NULL(output_);
    EvalOnRun();
    MS_LOG(DEBUG) << "Run result of [" << name() << "] : " << output_->ToString();
  }

  virtual bool EqualsTo(const OpPtr &other);

  const SymbolPtrList &inputs() const { return inputs_; }
  const SymbolPtr &input(size_t i) const {
    if (i >= inputs_.size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "The index " << i << " is out of range of the inputs size " << inputs_.size();
    }
    return inputs_[i];
  }
  size_t input_num() const { return inputs_.size(); }
  const SymbolPtr &output() const { return output_; }
  template <typename T>
  const T *input_as(size_t i) const {
    const T *p = input(i)->as<T>();
    if (p == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert failed for input " << i << " of " << name();
    }
    return p;
  }
  template <typename T>
  std::shared_ptr<T> input_as_sptr(size_t i) const {
    auto p = input(i)->as_sptr<T>();
    if (p == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert failed for input " << i << " of " << name();
    }
    return p;
  }
  template <typename T>
  T *output_as() const {
    if (output_ == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "The output of " << name() << " is not initialized.";
    }
    T *p = output_->as<T>();
    if (p == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert failed for output of " << name();
    }
    return p;
  }

  bool need_eval() const { return need_eval_; }

  virtual std::string name() const { return type_name(); }

  // overwrite shared_from_this to get OpPtr directly
  OpPtr shared_from_this() { return shared_from_base<Operation>(); }
  std::string ToString() const override;
  std::string DumpText() const override;

  class Emitter {
   public:
    explicit Emitter(OpPtrList *op_list = nullptr) : ops_(op_list) {}
    ~Emitter() = default;
    SymbolPtr Emit(const OpPtr &op) const;
    void Clean() { Emitter::cse_cache_.clear(); }

   private:
    void Cse(const OpPtr &op);
    static inline std::map<std::string, OpPtrList> cse_cache_;
    OpPtrList *ops_;
  };

 protected:
  virtual SymbolPtr Eval() = 0;
  virtual void EvalOnRun() { output_->Update(Eval()); }
  virtual void UpdateMathInfo() {}

  bool is_building() const { return is_building_; }
  void DoNotEvalOnRun() {
    if (is_building_) {
      need_eval_ = false;
    }
  }

  void SetEmitter(const Emitter *e) { emitter_ = e; }
  const Emitter &emitter() const {
    static Emitter e(nullptr);
    return emitter_ != nullptr ? *emitter_ : e;
  }
  SymbolPtr Emit(const OpPtr &op) const { return emitter().Emit(op); }

  SymbolPtr ResultIntList(SymbolPtrList &&result) {
    if (is_building()) {
      return GenList(result);
    }
    output_as<ListSymbol>()->UpdateList(result);
    return nullptr;
  }

  SymbolPtr GenInt(int64_t v) { return IntSymbol::Make(v, shared_from_this()); }
  SymbolPtr GenVInt() { return IntSymbol::Make(shared_from_this()); }
  SymbolPtr GenList(const SymbolPtrList &list) { return ListSymbol::Make(list, shared_from_this()); }
  SymbolPtr GenList(SymbolPtrList &&list) { return ListSymbol::Make(list, shared_from_this()); }
  SymbolPtr GenList(const std::initializer_list<SymbolPtr> &list) { return ListSymbol::Make(list, shared_from_this()); }
  SymbolPtr GenVList() { return ListSymbol::Make(shared_from_this()); }

  SymbolPtr GenVIntList(size_t n) {
    SymbolPtrList list(n);
    std::generate(list.begin(), list.end(), [this]() { return this->GenVInt(); });
    return GenList(std::move(list));
  }

  SymbolPtr output_{nullptr};
  bool support_commutative_law_{false};

 private:
  const Emitter *emitter_{nullptr};
  SymbolPtrList inputs_;
  bool is_building_{true};
  bool need_eval_{true};
};
using OperationEmitter = Operation::Emitter;
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_SYMBOLIC_SHAPE_OPERATION_H_
