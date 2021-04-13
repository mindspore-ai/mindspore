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

#ifndef MINDQUANTUM_ENGINE_CHAMILTONIAN_H_
#define MINDQUANTUM_ENGINE_CHAMILTONIAN_H_
#include "projectq/backends/_sim/_cppkernels/simulator.hpp"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/basic_gates.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/sparse.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"

namespace mindspore {
namespace mindquantum {
class Hamiltonian {
 private:
  sparse::GoodHamilt ham_;
  Index n_qubits_;
  sparse::DequeSparseHam ham_sparse_;
  Simulator::TermsDict td_;
  Simulator::ComplexTermsDict ctd_;
  int final_size_ = 1;
  bool ham_sparsed_ = false;

 public:
  Hamiltonian();
  Hamiltonian(const sparse::GoodHamilt &, Index);
  sparse::DequeSparseHam TransHamiltonianPhaseOne(int, const sparse::GoodHamilt &, Index);
  int TransHamiltonianPhaseTwo(sparse::DequeSparseHam &, int, int);
  void SparseHamiltonian(int, int, int);
  void SetTermsDict(Simulator::TermsDict const &);
  void Sparsed(bool);
  const Simulator::ComplexTermsDict &GetCTD() const;
  const Simulator::TermsDict &GetTD() const;
};
}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_CHAMILTONIAN_H_
