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
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/gates.h"

#include <cmath>

namespace mindspore {
namespace mindquantum {
RXGate::RXGate(const Indexes &obj_qubits, const Indexes &ctrl_qubits, const ParameterResolver &paras)
    : IntrinsicOneParaGate("RX", obj_qubits, ctrl_qubits, paras) {}

RXGate::RXGate() : IntrinsicOneParaGate("RX", {}, {}, {}) {}

Matrix RXGate::GetIntrinsicMatrix(CalcType theta) {
  Matrix result = {{{cos(theta / 2), 0}, {0, -sin(theta / 2)}}, {{0, -sin(theta / 2)}, {cos(theta / 2), 0}}};
  return result;
}

Matrix RXGate::GetIntrinsicDiffMatrix(CalcType theta) {
  Matrix result = {{{-sin(theta / 2) / 2, 0}, {0, -cos(theta / 2) / 2}},
                   {{0, -cos(theta / 2) / 2}, {-sin(theta / 2) / 2, 0}}};
  return result;
}

RYGate::RYGate(const Indexes &obj_qubits, const Indexes &ctrl_qubits, const ParameterResolver &paras)
    : IntrinsicOneParaGate("RY", obj_qubits, ctrl_qubits, paras) {}

Matrix RYGate::GetIntrinsicMatrix(CalcType theta) {
  Matrix result = {{{cos(theta / 2), 0}, {-sin(theta / 2), 0}}, {{sin(theta / 2), 0}, {cos(theta / 2), 0}}};
  return result;
}

Matrix RYGate::GetIntrinsicDiffMatrix(CalcType theta) {
  Matrix result = {{{-sin(theta / 2) / 2, 0}, {-cos(theta / 2) / 2, 0}},
                   {{cos(theta / 2) / 2, 0}, {-sin(theta / 2) / 2, 0}}};
  return result;
}

RZGate::RZGate(const Indexes &obj_qubits, const Indexes &ctrl_qubits, const ParameterResolver &paras)
    : IntrinsicOneParaGate("RZ", obj_qubits, ctrl_qubits, paras) {}

Matrix RZGate::GetIntrinsicMatrix(CalcType theta) {
  Matrix result = {{{cos(theta / 2), -sin(theta / 2)}, {0, 0}}, {{0, 0}, {cos(theta / 2), sin(theta / 2)}}};
  return result;
}

Matrix RZGate::GetIntrinsicDiffMatrix(CalcType theta) {
  Matrix result = {{{-sin(theta / 2) / 2, -cos(theta / 2) / 2}, {0, 0}},
                   {{0, 0}, {-sin(theta / 2) / 2, cos(theta / 2) / 2}}};
  return result;
}

PhaseShiftGate::PhaseShiftGate(const Indexes &obj_qubits, const Indexes &ctrl_qubits, const ParameterResolver &paras)
    : IntrinsicOneParaGate("PS", obj_qubits, ctrl_qubits, paras) {}

Matrix PhaseShiftGate::GetIntrinsicMatrix(CalcType theta) {
  Matrix result = {{{1, 0}, {0, 0}}, {{0, 0}, {cos(theta), sin(theta)}}};
  return result;
}

Matrix PhaseShiftGate::GetIntrinsicDiffMatrix(CalcType theta) {
  Matrix result = {{{0, 0}, {0, 0}}, {{0, 0}, {-sin(theta), cos(theta)}}};
  return result;
}

XXGate::XXGate(const Indexes &obj_qubits, const Indexes &ctrl_qubits, const ParameterResolver &paras)
    : IntrinsicOneParaGate("XX", obj_qubits, ctrl_qubits, paras) {}

Matrix XXGate::GetIntrinsicMatrix(CalcType theta) {
  double c = cos(theta);
  double s = sin(theta);

  Matrix result = {{{c, 0}, {0, 0}, {0, 0}, {0, -s}},
                   {{0, 0}, {c, 0}, {0, -s}, {0, 0}},
                   {{0, 0}, {0, -s}, {c, 0}, {0, 0}},
                   {{0, -s}, {0, 0}, {0, 0}, {c, 0}}};
  return result;
}

Matrix XXGate::GetIntrinsicDiffMatrix(CalcType theta) {
  double c = cos(theta);
  double s = sin(theta);

  Matrix result = {{{-s, 0}, {0, 0}, {0, 0}, {0, -c}},
                   {{0, 0}, {-s, 0}, {0, -c}, {0, 0}},
                   {{0, 0}, {0, -c}, {-s, 0}, {0, 0}},
                   {{0, -c}, {0, 0}, {0, 0}, {-s, 0}}};
  return result;
}

YYGate::YYGate(const Indexes &obj_qubits, const Indexes &ctrl_qubits, const ParameterResolver &paras)
    : IntrinsicOneParaGate("YY", obj_qubits, ctrl_qubits, paras) {}

Matrix YYGate::GetIntrinsicMatrix(CalcType theta) {
  double c = cos(theta);
  double s = sin(theta);

  Matrix result = {{{c, 0}, {0, 0}, {0, 0}, {0, s}},
                   {{0, 0}, {c, 0}, {0, -s}, {0, 0}},
                   {{0, 0}, {0, -s}, {c, 0}, {0, 0}},
                   {{0, s}, {0, 0}, {0, 0}, {c, 0}}};
  return result;
}

Matrix YYGate::GetIntrinsicDiffMatrix(CalcType theta) {
  double c = cos(theta);
  double s = sin(theta);

  Matrix result = {{{-s, 0}, {0, 0}, {0, 0}, {0, c}},
                   {{0, 0}, {-s, 0}, {0, -c}, {0, 0}},
                   {{0, 0}, {0, -c}, {-s, 0}, {0, 0}},
                   {{0, c}, {0, 0}, {0, 0}, {-s, 0}}};
  return result;
}

ZZGate::ZZGate(const Indexes &obj_qubits, const Indexes &ctrl_qubits, const ParameterResolver &paras)
    : IntrinsicOneParaGate("ZZ", obj_qubits, ctrl_qubits, paras) {}

Matrix ZZGate::GetIntrinsicMatrix(CalcType theta) {
  double c = cos(theta);
  double s = sin(theta);

  Matrix result = {{{c, -s}, {0, 0}, {0, 0}, {0, 0}},
                   {{0, 0}, {c, s}, {0, 0}, {0, 0}},
                   {{0, 0}, {0, 0}, {c, s}, {0, 0}},
                   {{0, 0}, {0, 0}, {0, 0}, {c, -s}}};
  return result;
}

Matrix ZZGate::GetIntrinsicDiffMatrix(CalcType theta) {
  double c = cos(theta);
  double s = sin(theta);

  Matrix result = {{{-s, -c}, {0, 0}, {0, 0}, {0, 0}},
                   {{0, 0}, {-s, c}, {0, 0}, {0, 0}},
                   {{0, 0}, {0, 0}, {-s, c}, {0, 0}},
                   {{0, 0}, {0, 0}, {0, 0}, {-s, -c}}};
  return result;
}
}  // namespace mindquantum
}  // namespace mindspore
