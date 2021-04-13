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
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/hamiltonian.h"

#include <utility>
namespace mindspore {
namespace mindquantum {
Hamiltonian::Hamiltonian() {}
Hamiltonian::Hamiltonian(const sparse::GoodHamilt &ham, Index n) : ham_(ham), n_qubits_(n) {}

sparse::DequeSparseHam Hamiltonian::TransHamiltonianPhaseOne(int n_thread1, const sparse::GoodHamilt &ham, Index n) {
  sparse::DequeSparseHam ham_sparse;
  ham_sparse.resize(ham.size());
  int step = 0;
#pragma omp parallel for schedule(static) num_threads(n_thread1)
  for (Index i = 0; i < ham.size(); i++) {
    auto &gt = ham.at(i);
    if (gt.second[0].first.size() == 0) {
      ham_sparse[i] = sparse::IdentitySparse(n) * gt.first.first * gt.second[0].second;
    } else {
      ham_sparse[i] = sparse::GoodTerm2Sparse(gt, n);
    }
    if ((++step) % 20 == 0) std::cout << "\r" << step << "\t/" << ham.size() << "\tfinshed" << std::flush;
  }
  std::cout << "\ncalculate hamiltonian phase1 finished\n";
  return ham_sparse;
}

int Hamiltonian::TransHamiltonianPhaseTwo(sparse::DequeSparseHam &ham_sparse, int n_thread2, int n_split) {
  int n = ham_sparse.size();
  while (n > 1) {
    int half = n / 2 + n % 2;
    std::cout << "n: " << n << "\t, half: " << half << "\n";
    if (n < n_split) {
      break;
    }
#pragma omp parallel for schedule(static) num_threads(half)
    for (int i = half; i < n; i++) {
      ham_sparse[i - half] += ham_sparse[i];
    }
    ham_sparse.erase(ham_sparse.end() - half + n % 2, ham_sparse.end());
    n = half;
  }
  std::cout << "total: " << ham_sparse.size() << " phase2 finished\n";
  return n;
}

void Hamiltonian::SparseHamiltonian(int n_thread1, int n_thread2, int n_split) {
  ham_sparse_ = Hamiltonian::TransHamiltonianPhaseOne(n_thread1, ham_, n_qubits_);
  final_size_ = Hamiltonian::TransHamiltonianPhaseTwo(ham_sparse_, n_thread2, n_split);
}

void Hamiltonian::SetTermsDict(Simulator::TermsDict const &d) {
  td_ = d;
  Simulator::ComplexTermsDict().swap(ctd_);
  for (auto &term : td_) {
    ComplexType coeff = {term.second, 0};
    ctd_.push_back(std::make_pair(term.first, coeff));
  }
}

void Hamiltonian::Sparsed(bool s) { ham_sparsed_ = s; }
const Simulator::ComplexTermsDict &Hamiltonian::GetCTD() const { return ctd_; }
const Simulator::TermsDict &Hamiltonian::GetTD() const { return td_; }
}  // namespace mindquantum
}  // namespace mindspore
