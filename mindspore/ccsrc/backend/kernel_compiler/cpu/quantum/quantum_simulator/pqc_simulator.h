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

#ifndef MINDQUANTUM_ENGINE_PQC_SIMULATOR_H_
#define MINDQUANTUM_ENGINE_PQC_SIMULATOR_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "projectq/backends/_sim/_cppkernels/simulator.hpp"
#include "utils/log_adapter.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/basic_gates.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/parameter_resolver.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/circuit.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/hamiltonian.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/transformer.h"

namespace mindspore {
namespace mindquantum {
struct CalcGradientParam {
  BasicCircuit *circuit_cp;
  BasicCircuit *circuit_hermitian_cp;
  transformer::Hamiltonians *hamiltonians_cp;
  ParameterResolver *paras_cp;
  transformer::NamesType *encoder_params_names_cp;
  transformer::NamesType *ansatz_params_names_cp;
  bool dummy_circuit_cp{false};
};
class PQCSimulator : public Simulator {
 private:
  Index n_qubits_;
  Indexes ordering_;
  Index len_;

 public:
  PQCSimulator();
  PQCSimulator(Index seed, Index N);
  void ApplyGate(std::shared_ptr<BasicGate>, const ParameterResolver &, bool);
  void ApplyBlock(const GateBlock &, const ParameterResolver &);
  void ApplyBlocks(const GateBlocks &, const ParameterResolver &);
  void Evolution(const BasicCircuit &, const ParameterResolver &);
  CalcType Measure(Index, Index, bool);
  void ApplyHamiltonian(const Hamiltonian &);
  CalcType GetExpectationValue(const Hamiltonian &);
  std::vector<std::vector<float>> CalcGradient(const std::shared_ptr<CalcGradientParam> &, PQCSimulator &,
                                               PQCSimulator &, PQCSimulator &);
  void AllocateAll();
  void DeallocateAll();
  void SetState(const StateVector &);
  std::size_t GetControlMask(Indexes const &);
  void SetZeroState();
};
}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_PQC_SIMULATOR_H_
