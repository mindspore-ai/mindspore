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
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/pqc_simulator.h"

#include <omp.h>
#include <numeric>

namespace mindspore {
namespace mindquantum {
PQCSimulator::PQCSimulator() : Simulator(1), n_qubits_(1) {
  PQCSimulator::AllocateAll();
  for (Index i = 0; i < n_qubits_; i++) {
    ordering_.push_back(i);
  }
  len_ = (1UL << n_qubits_);
}

PQCSimulator::PQCSimulator(Index seed = 1, Index N = 1) : Simulator(seed), n_qubits_(N) {
  PQCSimulator::AllocateAll();
  for (Index i = 0; i < n_qubits_; i++) {
    ordering_.push_back(i);
  }
  len_ = (1UL << n_qubits_);
}

void PQCSimulator::ApplyGate(std::shared_ptr<BasicGate> g, const ParameterResolver &paras, bool diff) {
  if (g->IsParameterGate()) {
    if (diff) {
      PQCSimulator::apply_controlled_gate(g->GetDiffMatrix(paras), g->GetObjQubits(), g->GetCtrlQubits());
    } else {
      PQCSimulator::apply_controlled_gate(g->GetMatrix(paras), g->GetObjQubits(), g->GetCtrlQubits());
    }
  } else {
    PQCSimulator::apply_controlled_gate(g->GetBaseMatrix(), g->GetObjQubits(), g->GetCtrlQubits());
  }
}

void PQCSimulator::ApplyBlock(const GateBlock &b, const mindquantum::ParameterResolver &paras) {
  for (auto &g : b) {
    PQCSimulator::ApplyGate(g, paras, false);
  }
  PQCSimulator::run();
}

void PQCSimulator::ApplyBlocks(const GateBlocks &bs, const ParameterResolver &paras) {
  for (auto &b : bs) {
    PQCSimulator::ApplyBlock(b, paras);
  }
}

void PQCSimulator::Evolution(BasicCircuit const &circuit, ParameterResolver const &paras) {
  PQCSimulator::ApplyBlocks(circuit.GetGateBlocks(), paras);
}

CalcType PQCSimulator::Measure(Index mask1, Index mask2, bool apply) {
  CalcType out = 0;
#pragma omp parallel for reduction(+ : out) schedule(static)
  for (unsigned i = 0; i < (1UL << n_qubits_); i++) {
    if (((i & mask1) == mask1) && ((i | mask2) == mask2)) {
      out = out + std::real(vec_[i]) * std::real(vec_[i]) + std::imag(vec_[i]) * std::imag(vec_[i]);
    } else if (apply) {
      vec_[i] = 0;
    }
  }
  return out;
}

std::vector<std::vector<float>> PQCSimulator::CalcGradient(const std::shared_ptr<CalcGradientParam> &input_params,
                                                           PQCSimulator &s_left, PQCSimulator &s_right,
                                                           PQCSimulator &s_right_tmp) {
  // Suppose the simulator already evaluate the circuit.
  auto circuit = input_params->circuit_cp;
  auto circuit_hermitian = input_params->circuit_hermitian_cp;
  auto hamiltonians = input_params->hamiltonians_cp;
  auto paras = input_params->paras_cp;
  auto encoder_params_names = input_params->encoder_params_names_cp;
  auto ansatz_params_names = input_params->ansatz_params_names_cp;
  auto dummy_circuit_ = input_params->dummy_circuit_cp;
  auto &circ_gate_blocks = circuit->GetGateBlocks();
  auto &circ_herm_gate_blocks = circuit_hermitian->GetGateBlocks();
  std::map<std::string, size_t> poi;
  for (size_t i = 0; i < encoder_params_names->size(); i++) {
    poi[encoder_params_names->at(i)] = i;
  }
  for (size_t i = 0; i < ansatz_params_names->size(); i++) {
    poi[ansatz_params_names->at(i)] = i + encoder_params_names->size();
  }
  if (circ_gate_blocks.size() == 0 || circ_herm_gate_blocks.size() == 0) {
    MS_LOG(EXCEPTION) << "Empty quantum circuit!";
  }
  unsigned len = circ_gate_blocks.at(0).size();
  std::vector<float> grad(hamiltonians->size() * poi.size(), 0);
  std::vector<float> e0(hamiltonians->size(), 0);

  // #pragma omp parallel for
  for (size_t h_index = 0; h_index < hamiltonians->size(); h_index++) {
    auto &hamiltonian = hamiltonians->at(h_index);
    s_right.set_wavefunction(vec_, ordering_);
    s_left.set_wavefunction(s_right.vec_, ordering_);
    s_left.apply_qubit_operator(hamiltonian.GetCTD(), ordering_);
    e0[h_index] = static_cast<float>(ComplexInnerProduct(vec_, s_left.vec_, len_).real());
    if (dummy_circuit_) {
      continue;
    }
    for (unsigned i = 0; i < len; i++) {
      if ((!circ_herm_gate_blocks.at(0)[i]->IsParameterGate()) ||
          (circ_herm_gate_blocks.at(0)[i]->GetParameterResolver().GetRequiresGradParameters().size() == 0)) {
        s_left.ApplyGate(circ_herm_gate_blocks.at(0)[i], *paras, false);
        s_right.ApplyGate(circ_herm_gate_blocks.at(0)[i], *paras, false);
      } else {
        s_right.ApplyGate(circ_herm_gate_blocks.at(0)[i], *paras, false);
        s_right.run();
        s_right_tmp.set_wavefunction(s_right.vec_, ordering_);
        s_right_tmp.ApplyGate(circ_gate_blocks.at(0)[len - 1 - i], *paras, true);
        s_right_tmp.run();
        s_left.run();
        ComplexType gi = 0;
        if (circ_herm_gate_blocks.at(0)[i]->GetCtrlQubits().size() == 0) {
          gi = ComplexInnerProduct(s_left.vec_, s_right_tmp.vec_, len_);
        } else {
          gi = ComplexInnerProductWithControl(s_left.vec_, s_right_tmp.vec_, len_,
                                              GetControlMask(circ_herm_gate_blocks.at(0)[i]->GetCtrlQubits()));
        }
        for (auto &it : circ_herm_gate_blocks.at(0)[i]->GetParameterResolver().GetRequiresGradParameters()) {
          grad[h_index * poi.size() + poi[it]] -= static_cast<float>(
            2 * circ_herm_gate_blocks.at(0)[i]->GetParameterResolver().GetData().at(it) * std::real(gi));
        }
        s_left.ApplyGate(circ_herm_gate_blocks.at(0)[i], *paras, false);
      }
    }
  }
  std::vector<float> grad1;
  std::vector<float> grad2;
  for (size_t i = 0; i < hamiltonians->size(); i++) {
    for (size_t j = 0; j < poi.size(); j++) {
      if (j < encoder_params_names->size()) {
        grad1.push_back(grad[i * poi.size() + j]);
      } else {
        grad2.push_back(grad[i * poi.size() + j]);
      }
    }
  }
  return {e0, grad1, grad2};
}

void PQCSimulator::AllocateAll() {
  for (unsigned i = 0; i < n_qubits_; i++) {
    Simulator::allocate_qubit(i);
  }
}
void PQCSimulator::DeallocateAll() {
  for (unsigned i = 0; i < n_qubits_; i++) {
    Simulator::deallocate_qubit(i);
  }
}
void PQCSimulator::SetState(const StateVector &wavefunction) { Simulator::set_wavefunction(wavefunction, ordering_); }

std::size_t PQCSimulator::GetControlMask(Indexes const &ctrls) {
  std::size_t ctrlmask =
    std::accumulate(ctrls.begin(), ctrls.end(), 0, [&](Index a, Index b) { return a | (1UL << ordering_[b]); });
  return ctrlmask;
}

void PQCSimulator::ApplyHamiltonian(const Hamiltonian &ham) {
  Simulator::apply_qubit_operator(ham.GetCTD(), ordering_);
}

CalcType PQCSimulator::GetExpectationValue(const Hamiltonian &ham) {
  return Simulator::get_expectation_value(ham.GetTD(), ordering_);
}
void PQCSimulator::SetZeroState() {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < len_; i++) {
    vec_[i] = {0, 0};
  }
  vec_[0] = {1, 0};
}
}  // namespace mindquantum
}  // namespace mindspore
