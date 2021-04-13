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

#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/circuit.h"

namespace mindspore {
namespace mindquantum {
BasicCircuit::BasicCircuit() : gate_blocks_({}) {}

void BasicCircuit::AppendBlock() { gate_blocks_.push_back({}); }

void BasicCircuit::AppendNoneParameterGate(const std::string &name, Matrix m, Indexes obj_qubits, Indexes ctrl_qubits) {
  auto npg = std::make_shared<NoneParameterGate>(name, m, obj_qubits, ctrl_qubits);
  gate_blocks_.back().push_back(npg);
}

void BasicCircuit::AppendParameterGate(const std::string &name, Indexes obj_qubits, Indexes ctrl_qubits,
                                       const ParameterResolver &paras) {
  if (name == "RX") {
    auto pg_rx = std::make_shared<RXGate>(obj_qubits, ctrl_qubits, paras);
    gate_blocks_.back().push_back(pg_rx);
  } else if (name == "RY") {
    auto pg_ry = std::make_shared<RYGate>(obj_qubits, ctrl_qubits, paras);
    gate_blocks_.back().push_back(pg_ry);
  } else if (name == "RZ") {
    auto pg_rz = std::make_shared<RZGate>(obj_qubits, ctrl_qubits, paras);
    gate_blocks_.back().push_back(pg_rz);
  } else if (name == "XX") {
    auto pg_xx = std::make_shared<XXGate>(obj_qubits, ctrl_qubits, paras);
    gate_blocks_.back().push_back(pg_xx);
  } else if (name == "YY") {
    auto pg_yy = std::make_shared<YYGate>(obj_qubits, ctrl_qubits, paras);
    gate_blocks_.back().push_back(pg_yy);
  } else if (name == "ZZ") {
    auto pg_zz = std::make_shared<ZZGate>(obj_qubits, ctrl_qubits, paras);
    gate_blocks_.back().push_back(pg_zz);
  } else if (name == "PS") {
    auto pg_ps = std::make_shared<PhaseShiftGate>(obj_qubits, ctrl_qubits, paras);
    gate_blocks_.back().push_back(pg_ps);
  } else {
  }
}

const GateBlocks &BasicCircuit::GetGateBlocks() const { return gate_blocks_; }
}  // namespace mindquantum
}  // namespace mindspore
