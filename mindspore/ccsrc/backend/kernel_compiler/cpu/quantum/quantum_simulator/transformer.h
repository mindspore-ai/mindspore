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

#ifndef MINDQUANTUM_ENGINE_TRANSFORMER_H_
#define MINDQUANTUM_ENGINE_TRANSFORMER_H_
#include <vector>
#include <string>
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/gates.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/circuit.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/parameter_resolver.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/hamiltonian.h"

namespace mindspore {
namespace mindquantum {
namespace transformer {
using NameType = std::string;
using MatrixColumnType = std::vector<std::string>;
using MatrixType = std::vector<MatrixColumnType>;
using ComplexMatrixType = std::vector<MatrixType>;
using ParaNameType = std::vector<std::string>;
using CoeffType = std::vector<float>;
using RequireType = std::vector<bool>;
using NamesType = std::vector<NameType>;
using ComplexMatrixsType = std::vector<ComplexMatrixType>;
using ParasNameType = std::vector<ParaNameType>;
using CoeffsType = std::vector<CoeffType>;
using RequiresType = std::vector<RequireType>;
using Indexess = std::vector<std::vector<int64_t>>;
using PauliCoeffsType = std::vector<float>;
using PaulisCoeffsType = std::vector<PauliCoeffsType>;
using PauliWordType = std::vector<std::string>;
using PauliWordsType = std::vector<PauliWordType>;
using PaulisWordsType = std::vector<PauliWordsType>;
using PauliQubitType = std::vector<int64_t>;
using PauliQubitsType = std::vector<PauliQubitType>;
using PaulisQubitsType = std::vector<PauliQubitsType>;
using Hamiltonians = std::vector<Hamiltonian>;

Hamiltonians HamiltoniansTransfor(const PaulisCoeffsType &, const PaulisWordsType &, const PaulisQubitsType &);

std::vector<BasicCircuit> CircuitTransfor(const NamesType &, const ComplexMatrixsType &, const Indexess &,
                                          const Indexess &, const ParasNameType &, const CoeffsType &,
                                          const RequiresType &);

Matrix MatrixConverter(const MatrixType &, const MatrixType &, bool);

ParameterResolver ParameterResolverConverter(const ParaNameType &, const CoeffType &, const RequireType &, bool);
}  // namespace transformer
}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_TRANSFORMER_H_
