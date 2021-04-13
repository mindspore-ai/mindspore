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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PQC_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PQC_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/pqc_simulator.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/transformer.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/circuit.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/parameter_resolver.h"

namespace mindspore {
namespace kernel {
class PQCCPUKernel : public CPUKernel {
 public:
  PQCCPUKernel() = default;
  ~PQCCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  void InitPQCStructure(const CNodePtr &kernel_node);

 private:
  size_t n_samples_;
  size_t n_threads_user_;
  bool dummy_circuit_;
  size_t result_len_;
  size_t encoder_g_len_;
  size_t ansatz_g_len_;

  int64_t n_qubits_;
  mindquantum::BasicCircuit circ_;
  mindquantum::BasicCircuit herm_circ_;
  mindquantum::transformer::Hamiltonians hams_;
  std::vector<std::vector<std::shared_ptr<mindquantum::PQCSimulator>>> tmp_sims_;

  // parameters
  mindquantum::transformer::NamesType encoder_params_names_;
  mindquantum::transformer::NamesType ansatz_params_names_;

  // quantum circuit
  mindquantum::transformer::NamesType gate_names_;
  mindquantum::transformer::ComplexMatrixsType gate_matrix_;
  mindquantum::transformer::Indexess gate_obj_qubits_;
  mindquantum::transformer::Indexess gate_ctrl_qubits_;
  mindquantum::transformer::ParasNameType gate_params_names_;
  mindquantum::transformer::CoeffsType gate_coeff_;
  mindquantum::transformer::RequiresType gate_requires_grad_;

  // hamiltonian
  mindquantum::transformer::PaulisCoeffsType hams_pauli_coeff_;
  mindquantum::transformer::PaulisWordsType hams_pauli_word_;
  mindquantum::transformer::PaulisQubitsType hams_pauli_qubit_;
};

MS_REG_CPU_KERNEL(PQC,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  PQCCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PQC_CPU_KERNEL_H_
