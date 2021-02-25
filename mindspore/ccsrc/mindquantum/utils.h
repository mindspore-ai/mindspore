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

#ifndef MINDQUANTUM_ENGINE_UTILS_H_
#define MINDQUANTUM_ENGINE_UTILS_H_
#include <string>
#include <complex>
#include <vector>
#include <map>
#include <set>
#include "projectq/backends/_sim/_cppkernels/intrin/alignedallocator.hpp"
#include "projectq/backends/_sim/_cppkernels/simulator.hpp"

namespace mindspore {
namespace mindquantum {
using CalcType = double;
using ComplexType = std::complex<CalcType>;
using ParaType = std::map<std::string, CalcType>;
using ParaSetType = std::set<std::string>;
using Matrix = std::vector<std::vector<ComplexType, aligned_allocator<ComplexType, 64>>>;
using Index = unsigned;
using Indexes = std::vector<Index>;
using ParaMapType = std::map<std::string, CalcType>;
ComplexType ComplexInnerProduct(const Simulator::StateVector &, const Simulator::StateVector &, Index);
ComplexType ComplexInnerProductWithControl(const Simulator::StateVector &, const Simulator::StateVector &, Index,
                                           std::size_t);
const char kNThreads[] = "n_threads";
const char kNQubits[] = "n_qubits";
const char kParamNames[] = "param_names";
const char kEncoderParamsNames[] = "encoder_params_names";
const char kAnsatzParamsNames[] = "ansatz_params_names";
const char kGateNames[] = "gate_names";
const char kGateMatrix[] = "gate_matrix";
const char kGateObjQubits[] = "gate_obj_qubits";
const char kGateCtrlQubits[] = "gate_ctrl_qubits";
const char kGateParamsNames[] = "gate_params_names";
const char kGateCoeff[] = "gate_coeff";
const char kGateRequiresGrad[] = "gate_requires_grad";
const char kHamsPauliCoeff[] = "hams_pauli_coeff";
const char kHamsPauliWord[] = "hams_pauli_word";
const char kHamsPauliQubit[] = "hams_pauli_qubit";
}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_UTILS_H_
