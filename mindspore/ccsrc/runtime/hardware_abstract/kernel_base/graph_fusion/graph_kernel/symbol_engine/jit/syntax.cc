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

#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/symbol_engine/jit/syntax.h"
#include "utils/log_adapter.h"

namespace mindspore::graphkernel::symshape::ast {
std::string BinOpType2Str(BinOpType tag) {
  switch (tag) {
    case BinOpType::ScalarAdd:
      return "ScalarAdd";
    case BinOpType::ScalarMul:
      return "ScalarMul";
    case BinOpType::ScalarSub:
      return "ScalarSub";
    case BinOpType::ScalarDiv:
      return "ScalarDiv";
    case BinOpType::ScalarMax:
      return "ScalarMax";
    case BinOpType::ScalarMin:
      return "ScalarMin";
    default:
      MS_LOG(EXCEPTION) << "Unexpected Behavior: BinOpType not recognized";
  }
  MS_LOG_EXCEPTION << "Unexpected Behavior: BinOpType not recognized";
}

void IntImm::Accept(Visitor *visitor) { visitor->Visit(*this); }
std::string IntImm::ToString() const {
  std::stringstream res;
  res << shape_int;
  return res.str();
}

void Var::Accept(Visitor *visitor) { visitor->Visit(*this); }
std::string Var::ToString() const { return name_; }

std::string BinOp::ToString() const {
  std::stringstream res;
  res << BinOpType2Str(optype_) << " ( " << a_->ToString() << " ," << b_->ToString() << " ) ";
  return res.str();
}

std::string Input::ToString() const {
  std::stringstream res;
  res << "Input " << i_ << " " << j_;
  return res.str();
}

void BinOp::Accept(Visitor *visitor) { visitor->Visit(*this); }

void Input::Accept(Visitor *visitor) { visitor->Visit(*this); }

void Shape::Accept(Visitor *visitor) { visitor->Visit(*this); }

std::string Shape::ToString() const {
  std::stringstream res;
  res << "[";
  for (auto smbl : this->smbls_) {
    res << smbl->ToString() << ",";
  }
  if (!smbls_.empty()) {
    res.seekp(-1, res.cur);
  }

  res << "]";
  return res.str();
}
}  // namespace mindspore::graphkernel::symshape::ast
