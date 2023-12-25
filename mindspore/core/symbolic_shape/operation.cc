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
#include "mindspore/core/symbolic_shape/operation.h"
#include "mindspore/core/symbolic_shape/utils.h"
#include "mindspore/core/ops/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
bool Operation::Build() {
  MS_EXCEPTION_IF_CHECK_FAIL(output_ == nullptr, "The operation is built.");
  MS_LOG(DEBUG) << "Building operation " << ToString();
  output_ = Eval();
  if (output_ == nullptr) {
    return false;
  }
  if (!output_->CanUpdate()) {
    need_eval_ = false;
  }
  UpdateMathInfo();
  MS_LOG(DEBUG) << "Build result of [" << name() << "]: " << output_->ToString() << ". need_eval=" << need_eval();
  is_building_ = false;
  return true;
}

bool Operation::EqualsTo(const OpPtr &other) {
  if (name() != other->name()) {
    return false;
  }
  if (inputs_.size() != other->inputs_.size()) {
    return false;
  }
  bool result = true;
  for (size_t i = 0; i < inputs_.size(); i++) {
    if (!inputs_[i]->EqualsTo(other->inputs_[i])) {
      result = false;
      break;
    }
  }
  if (!result && inputs_.size() == 2 && support_commutative_law_) {
    return inputs_[0]->EqualsTo(other->inputs_[1]) && inputs_[1]->EqualsTo(other->inputs_[0]);
  }
  return result;
}

std::string Operation::ToString() const { return name() + SymbolListToStr(inputs(), "(", ")"); }

std::string Operation::DumpText() const {
  std::ostringstream oss;
  MS_EXCEPTION_IF_NULL(output_);
  oss << output_->ToString() << " = " << this->ToString() << "\n";
  return oss.str();
}

SymbolPtr Operation::Emitter::Emit(const OpPtr &op) const {
  op->SetEmitter(this);
  auto result = op->Build();
  if (!result) {
    MS_LOG(INFO) << "Failed to build op " << op->ToString();
    return nullptr;
  }
  op->SetEmitter(nullptr);
  if (ops_ != nullptr && op->need_eval()) {
    const_cast<Emitter *>(this)->Cse(op);
    (void)ops_->emplace_back(op);
  }
  return op->output();
}

void Operation::Emitter::Cse(const OpPtr &op) {
  if (!op->isa<ops::ScalarOp>()) {
    return;
  }
  bool has_same = false;
  for (auto &prev_op : cse_cache_[op->name()]) {
    if (op->EqualsTo(prev_op)) {
      MS_LOG(DEBUG) << "The op \"" << op->output()->ToString() << "=" << op->ToString() << "\" is same as prev_op \""
                    << prev_op->output()->ToString() << "=" << prev_op->ToString() << "\"";
      prev_op->output_as<IntSymbol>()->SetEqual(op->output()->as_sptr<IntSymbol>());
      has_same = true;
      break;
    }
  }
  if (!has_same) {
    (void)cse_cache_[op->name()].emplace_back(op);
  }
}
}  // namespace symshape
}  // namespace mindspore
