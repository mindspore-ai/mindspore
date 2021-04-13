/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDQUANTUM_ENGINE_GATES_H_
#define MINDQUANTUM_ENGINE_GATES_H_
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/intrinsic_one_para_gate.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"

namespace mindspore {
namespace mindquantum {
class RXGate : public IntrinsicOneParaGate {
  Matrix GetIntrinsicMatrix(CalcType) override;
  Matrix GetIntrinsicDiffMatrix(CalcType) override;

 public:
  RXGate(const Indexes &, const Indexes &, const ParameterResolver &);
  RXGate();
};

class RYGate : public IntrinsicOneParaGate {
  Matrix GetIntrinsicMatrix(CalcType) override;
  Matrix GetIntrinsicDiffMatrix(CalcType) override;

 public:
  RYGate(const Indexes &, const Indexes &, const ParameterResolver &);
};

class RZGate : public IntrinsicOneParaGate {
  Matrix GetIntrinsicMatrix(CalcType) override;
  Matrix GetIntrinsicDiffMatrix(CalcType) override;

 public:
  RZGate(const Indexes &, const Indexes &, const ParameterResolver &);
};

class PhaseShiftGate : public IntrinsicOneParaGate {
  Matrix GetIntrinsicMatrix(CalcType) override;
  Matrix GetIntrinsicDiffMatrix(CalcType) override;

 public:
  PhaseShiftGate(const Indexes &, const Indexes &, const ParameterResolver &);
};

class XXGate : public IntrinsicOneParaGate {
  Matrix GetIntrinsicMatrix(CalcType) override;
  Matrix GetIntrinsicDiffMatrix(CalcType) override;

 public:
  XXGate(const Indexes &, const Indexes &, const ParameterResolver &);
};

class YYGate : public IntrinsicOneParaGate {
  Matrix GetIntrinsicMatrix(CalcType) override;
  Matrix GetIntrinsicDiffMatrix(CalcType) override;

 public:
  YYGate(const Indexes &, const Indexes &, const ParameterResolver &);
};

class ZZGate : public IntrinsicOneParaGate {
  Matrix GetIntrinsicMatrix(CalcType) override;
  Matrix GetIntrinsicDiffMatrix(CalcType) override;

 public:
  ZZGate(const Indexes &, const Indexes &, const ParameterResolver &);
};

}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_GATES_H_
