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
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/transformer.h"

#include <algorithm>
#include <utility>

namespace mindspore {
namespace mindquantum {
namespace transformer {
Matrix MatrixConverter(const MatrixType &matrix_real, const MatrixType &matrix_imag, bool hermitian) {
  Matrix out;
  for (Index i = 0; i < matrix_real.size(); i++) {
    out.push_back({});
    for (Index j = 0; j < matrix_real.size(); j++) {
      if (hermitian)
        out.back().push_back({stod(matrix_real[j][i]), -stod(matrix_imag[j][i])});
      else
        out.back().push_back({stod(matrix_real[i][j]), stod(matrix_imag[i][j])});
    }
  }
  return out;
}

ParameterResolver ParameterResolverConverter(const ParaNameType &para_name, const CoeffType &coeff,
                                             const RequireType &require_grad, bool hermitian) {
  ParameterResolver pr;
  for (Index i = 0; i < para_name.size(); i++) {
    auto name = para_name[i];
    if (hermitian)
      pr.SetData(name, -coeff[i]);
    else
      pr.SetData(name, coeff[i]);
    if (require_grad[i])
      pr.InsertRequiresGrad(name);
    else
      pr.InsertNoGrad(name);
  }
  return pr;
}

std::vector<BasicCircuit> CircuitTransfor(const NamesType &names, const ComplexMatrixsType &matrixs,
                                          const Indexess &objs_qubits, const Indexess &ctrls_qubits,
                                          const ParasNameType &paras_name, const CoeffsType &coeffs,
                                          const RequiresType &requires_grad) {
  BasicCircuit circuit = BasicCircuit();
  BasicCircuit herm_circuit = BasicCircuit();
  circuit.AppendBlock();
  herm_circuit.AppendBlock();
  for (Index n = 0; n < names.size(); n++) {
    Indexes obj(objs_qubits[n].size());
    Indexes ctrl(ctrls_qubits[n].size());
    std::transform(objs_qubits[n].begin(), objs_qubits[n].end(), obj.begin(),
                   [](const int64_t &i) { return (Index)(i); });
    std::transform(ctrls_qubits[n].begin(), ctrls_qubits[n].end(), ctrl.begin(),
                   [](const int64_t &i) { return (Index)(i); });
    if (names[n] == "npg")
      // non parameterize gate
      circuit.AppendNoneParameterGate("npg", MatrixConverter(matrixs[n][0], matrixs[n][1], false), obj, ctrl);
    else
      circuit.AppendParameterGate(names[n], obj, ctrl,
                                  ParameterResolverConverter(paras_name[n], coeffs[n], requires_grad[n], false));
  }
  for (Index n = 0; n < names.size(); n++) {
    Index tail = names.size() - 1 - n;
    Indexes obj(objs_qubits[tail].size());
    Indexes ctrl(ctrls_qubits[tail].size());
    std::transform(objs_qubits[tail].begin(), objs_qubits[tail].end(), obj.begin(),
                   [](const int64_t &i) { return (Index)(i); });
    std::transform(ctrls_qubits[tail].begin(), ctrls_qubits[tail].end(), ctrl.begin(),
                   [](const int64_t &i) { return (Index)(i); });
    if (names[tail] == "npg")
      // non parameterize gate
      herm_circuit.AppendNoneParameterGate("npg", MatrixConverter(matrixs[tail][0], matrixs[tail][1], true), obj, ctrl);
    else
      herm_circuit.AppendParameterGate(
        names[tail], obj, ctrl, ParameterResolverConverter(paras_name[tail], coeffs[tail], requires_grad[tail], true));
  }
  return {circuit, herm_circuit};
}

Hamiltonians HamiltoniansTransfor(const PaulisCoeffsType &paulis_coeffs, const PaulisWordsType &paulis_words,
                                  const PaulisQubitsType &paulis_qubits) {
  Hamiltonians hams;
  for (Index n = 0; n < paulis_coeffs.size(); n++) {
    Hamiltonian ham;
    Simulator::TermsDict td;
    for (Index i = 0; i < paulis_coeffs[n].size(); i++) {
      Simulator::Term term;
      for (Index j = 0; j < paulis_words[n][i].size(); j++)
        term.push_back(std::make_pair((Index)(paulis_qubits[n][i][j]), paulis_words[n][i][j].at(0)));
      td.push_back(std::make_pair(term, paulis_coeffs[n][i]));
    }
    ham.SetTermsDict(td);
    hams.push_back(ham);
  }
  return hams;
}

}  // namespace transformer
}  // namespace mindquantum
}  // namespace mindspore
