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
#ifndef MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_OPERATION_H_
#define MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_OPERATION_H_
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <map>

#include "abstract/symbolic_shape/symbol.h"
#include "abstract/symbolic_shape/int_symbol.h"

namespace mindspore {
namespace symshape {
/// \brief Operation is the basic class of operators for symbol.
class MS_CORE_API Operation : public Base {
 public:
  explicit Operation(SymbolPtrList &&inputs) : inputs_(std::move(inputs)) {}
  virtual ~Operation() = default;
  MS_DECLARE_PARENT(Operation, Base)

  void Build();
  inline void Run() {
    MS_LOG(DEBUG) << "Running operation " << ToString();
    MS_EXCEPTION_IF_NULL(output_);
    EvalOnRun();
    MS_LOG(DEBUG) << "Run result of [" << name() << "] : " << output_->ToString();
  }

  virtual bool EqualsTo(const OpPtr &other);
  virtual bool SupportCse() const { return false; }

  const SymbolPtrList &inputs() const { return inputs_; }
  const SymbolPtr &input(size_t i) const {
    if (MS_UNLIKELY(i >= inputs_.size())) {
      MS_LOG(INTERNAL_EXCEPTION) << "The index " << i << " is out of range of the inputs size " << inputs_.size();
    }
    return inputs_[i];
  }
  size_t input_num() const { return inputs_.size(); }
  const SymbolPtr &output() const { return output_; }
  template <typename T>
  const T *input_as(size_t i) const {
    const T *p = input(i)->as_noexcept<T>();
    if (MS_UNLIKELY(p == nullptr)) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert failed for input " << i << " of " << name();
    }
    return p;
  }
  template <typename T>
  std::shared_ptr<T> input_as_sptr(size_t i) const {
    auto p = input(i)->as_sptr_noexcept<T>();
    if (MS_UNLIKELY(p == nullptr)) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert failed for input " << i << " of " << name();
    }
    return p;
  }
  template <typename T>
  T *output_as() const {
    if (MS_UNLIKELY(output_ == nullptr)) {
      MS_LOG(INTERNAL_EXCEPTION) << "The output of " << name() << " is not initialized.";
    }
    T *p = output_->as_noexcept<T>();
    if (MS_UNLIKELY(p == nullptr)) {
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

  class MS_CORE_API Emitter {
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
  friend class OperationBuilder;

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
      return GenList(std::move(result));
    }
    output_as<ListSymbol>()->UpdateList(std::move(result));
    return nullptr;
  }

  SymbolPtr GenInt(int64_t v) { return IntSymbol::Make(v, shared_from_this()); }
  SymbolPtr GenVInt() { return IntSymbol::Make(shared_from_this()); }
  SymbolPtr GenList(const SymbolPtrList &list) { return ListSymbol::Make(list, shared_from_this()); }
  SymbolPtr GenList(SymbolPtrList &&list) { return ListSymbol::Make(std::move(list), shared_from_this()); }
  SymbolPtr GenList(const std::initializer_list<SymbolPtr> &list) { return ListSymbol::Make(list, shared_from_this()); }
  SymbolPtr GenVList() { return ListSymbol::Make(shared_from_this()); }

  SymbolPtr GenVIntList(size_t n) {
    SymbolPtrList list(n);
    std::generate(list.begin(), list.end(), [this]() { return this->GenVInt(); });
    return GenList(std::move(list));
  }

  // output abstract only can be used on building
  void SetOutAbstract(const abstract::AbstractBasePtr &abs) { out_abstract_ = abs; }
  const abstract::AbstractBasePtr &out_abstract() const {
    MS_EXCEPTION_IF_NULL(out_abstract_);
    return out_abstract_;
  }

  SymbolPtr output_{nullptr};
  bool support_commutative_law_{false};

 private:
  const Emitter *emitter_{nullptr};
  SymbolPtrList inputs_;
  bool is_building_{true};
  bool need_eval_{true};
  abstract::AbstractBasePtr out_abstract_{nullptr};
};
using OperationEmitter = Operation::Emitter;
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_OPERATION_H_
