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

#ifndef MINDQUANTUM_ENGINE_SPARSE_H_
#define MINDQUANTUM_ENGINE_SPARSE_H_
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>
#include <deque>
#include <complex>
#include <utility>
#include <iostream>
#include <vector>
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"

namespace mindspore {
namespace mindquantum {
namespace sparse {
using PauliWord = std::pair<Index, char>;
using PauliTerm = std::pair<std::vector<PauliWord>, int>;
using GoodTerm = std::pair<std::pair<CalcType, std::pair<Index, Index>>, std::vector<PauliTerm>>;
using GoodHamilt = std::vector<GoodTerm>;
typedef Eigen::VectorXcd EigenComplexVector;
using Eigen::VectorXi;
typedef Eigen::SparseMatrix<ComplexType, Eigen::RowMajor, int64_t> SparseMatrix;
using DequeSparseHam = std::deque<SparseMatrix>;
using KroneckerProductSparse = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>;
SparseMatrix BasiGateSparse(char);
SparseMatrix IdentitySparse(int);
SparseMatrix PauliTerm2Sparse(const PauliTerm &, Index, Index);
SparseMatrix GoodTerm2Sparse(const GoodTerm &, Index);
}  // namespace sparse
}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_SPARSE_H_
