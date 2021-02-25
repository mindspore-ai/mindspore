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
#include "backend/kernel_compiler/cpu/quantum/evolution_cpu_kernel.h"

#include <memory>
#include <algorithm>
#include "utils/ms_utils.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void EvolutionCPUKernel::InitPQCStructure(const CNodePtr &kernel_node) {
  n_qubits_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, mindquantum::kNQubits);
  param_names_ = AnfAlgo::GetNodeAttr<mindquantum::transformer::NamesType>(kernel_node, mindquantum::kParamNames);
  gate_names_ = AnfAlgo::GetNodeAttr<mindquantum::transformer::NamesType>(kernel_node, mindquantum::kGateNames);
  gate_matrix_ =
    AnfAlgo::GetNodeAttr<mindquantum::transformer::ComplexMatrixsType>(kernel_node, mindquantum::kGateMatrix);
  gate_obj_qubits_ = AnfAlgo::GetNodeAttr<mindquantum::transformer::Indexess>(kernel_node, mindquantum::kGateObjQubits);
  gate_ctrl_qubits_ =
    AnfAlgo::GetNodeAttr<mindquantum::transformer::Indexess>(kernel_node, mindquantum::kGateCtrlQubits);
  gate_params_names_ =
    AnfAlgo::GetNodeAttr<mindquantum::transformer::ParasNameType>(kernel_node, mindquantum::kGateParamsNames);
  gate_coeff_ = AnfAlgo::GetNodeAttr<mindquantum::transformer::CoeffsType>(kernel_node, mindquantum::kGateCoeff);
  gate_requires_grad_ =
    AnfAlgo::GetNodeAttr<mindquantum::transformer::RequiresType>(kernel_node, mindquantum::kGateRequiresGrad);
  hams_pauli_coeff_ =
    AnfAlgo::GetNodeAttr<mindquantum::transformer::PaulisCoeffsType>(kernel_node, mindquantum::kHamsPauliCoeff);
  hams_pauli_word_ =
    AnfAlgo::GetNodeAttr<mindquantum::transformer::PaulisWordsType>(kernel_node, mindquantum::kHamsPauliWord);
  hams_pauli_qubit_ =
    AnfAlgo::GetNodeAttr<mindquantum::transformer::PaulisQubitsType>(kernel_node, mindquantum::kHamsPauliQubit);
}

void EvolutionCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> param_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> result_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (param_shape.size() != 1 || result_shape.size() != 2) {
    MS_LOG(EXCEPTION) << "evolution invalid input size";
  }
  state_len_ = result_shape[0];
  InitPQCStructure(kernel_node);
  auto circs = mindquantum::transformer::CircuitTransfor(gate_names_, gate_matrix_, gate_obj_qubits_, gate_ctrl_qubits_,
                                                         gate_params_names_, gate_coeff_, gate_requires_grad_);
  circ_ = circs[0];
  hams_ = mindquantum::transformer::HamiltoniansTransfor(hams_pauli_coeff_, hams_pauli_word_, hams_pauli_qubit_);
  if (hams_.size() > 1) {
    MS_LOG(EXCEPTION) << "evolution only work for single hamiltonian or no hamiltonian.";
  }
}

bool EvolutionCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspace*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "evolution error input output size!";
  }
  auto param_data = reinterpret_cast<float *>(inputs[0]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(param_data);
  MS_EXCEPTION_IF_NULL(output);
  sim_ = mindquantum::PQCSimulator(1, n_qubits_);
  mindquantum::ParameterResolver pr;
  for (size_t i = 0; i < param_names_.size(); i++) {
    pr.SetData(param_names_.at(i), param_data[i]);
  }
  sim_.Evolution(circ_, pr);
  if (hams_.size() == 1) {
    sim_.ApplyHamiltonian(hams_[0]);
  }
  if (state_len_ != sim_.vec_.size()) {
    MS_LOG(EXCEPTION) << "simulation error number of quantum qubit!";
  }
  size_t poi = 0;
  for (auto &v : sim_.vec_) {
    output[poi++] = v.real();
    output[poi++] = v.imag();
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
