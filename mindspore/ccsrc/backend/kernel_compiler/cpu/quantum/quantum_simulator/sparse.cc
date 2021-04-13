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

#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/sparse.h"

namespace mindspore {
namespace mindquantum {
namespace sparse {
SparseMatrix BasiGateSparse(char g) {
  SparseMatrix out(2, 2);
  out.reserve(VectorXi::Constant(2, 2));
  switch (g) {
    case 'X':
    case 'x':
      out.insert(0, 1) = 1;
      out.insert(1, 0) = 1;
      break;

    case 'Y':
    case 'y':
      out.insert(0, 1) = {0, -1};
      out.insert(1, 0) = {0, 1};
      break;

    case 'Z':
    case 'z':
      out.insert(0, 0) = 1;
      out.insert(1, 1) = -1;
      break;

    case '0':
      out.insert(0, 0) = 1;
      break;

    case '1':
      out.insert(1, 1) = 1;
      break;

    default:
      out.insert(0, 0) = 1;
      out.insert(1, 1) = 1;
      break;
  }

  out.makeCompressed();

  return out;
}

SparseMatrix IdentitySparse(int n_qubit) {
  if (n_qubit == 0) {
    int dim = 1UL << n_qubit;
    SparseMatrix out(dim, dim);
    out.reserve(VectorXi::Constant(dim, dim));
    for (int i = 0; i < dim; i++) {
      out.insert(i, i) = 1;
    }
    out.makeCompressed();
    return out;
  } else {
    SparseMatrix out = BasiGateSparse('I');
    for (int i = 1; i < n_qubit; i++) {
      out = KroneckerProductSparse(out, BasiGateSparse('I')).eval();
    }
    return out;
  }
}

SparseMatrix PauliTerm2Sparse(const PauliTerm &pt, Index _min, Index _max) {
  int poi;
  int n = pt.first.size();
  SparseMatrix out;
  if (pt.first[0].first == _min) {
    out = BasiGateSparse(pt.first[0].second) * pt.second;
    poi = 1;
  } else {
    out = BasiGateSparse('I') * pt.second;
    poi = 0;
  }

  for (Index i = _min + 1; i <= _max; i++) {
    if (poi == n) {
      out = KroneckerProductSparse(IdentitySparse(_max - i + 1), out).eval();
      break;
    } else {
      if (i == pt.first[poi].first) {
        out = KroneckerProductSparse(BasiGateSparse(pt.first[poi++].second), out).eval();
      } else {
        out = KroneckerProductSparse(BasiGateSparse('I'), out).eval();
      }
    }
  }
  return out;
}

SparseMatrix GoodTerm2Sparse(const GoodTerm &gt, Index n_qubits) {
  SparseMatrix out = PauliTerm2Sparse(gt.second[0], gt.first.second.first, gt.first.second.second);
  for (Index i = 1; i < gt.second.size(); i++) {
    out += PauliTerm2Sparse(gt.second[i], gt.first.second.first, gt.first.second.second);
  }
  out.prune({0.0, 0.0});

  out *= gt.first.first;
  out = KroneckerProductSparse(out, IdentitySparse(gt.first.second.first)).eval();
  out = KroneckerProductSparse(IdentitySparse(n_qubits - gt.first.second.second - 1), out).eval();
  return out;
}

}  // namespace sparse
}  // namespace mindquantum
}  // namespace mindspore
