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
#include "backend/kernel_compiler/cpu/quantum/pqc_cpu_kernel.h"

#include <omp.h>
#include <utility>
#include <thread>
#include <memory>
#include <algorithm>
#include "utils/ms_utils.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
struct ComputeParam {
  float *encoder_data_cp{nullptr};
  float *ansatz_data_cp{nullptr};
  float *output_cp{nullptr};
  float *gradient_encoder_cp{nullptr};
  float *gradient_ansatz_cp{nullptr};
  mindquantum::BasicCircuit *circ_cp;
  mindquantum::BasicCircuit *herm_circ_cp;
  mindquantum::transformer::Hamiltonians *hams_cp;
  mindquantum::transformer::NamesType *encoder_params_names_cp;
  mindquantum::transformer::NamesType *ansatz_params_names_cp;
  std::vector<std::vector<std::shared_ptr<mindquantum::PQCSimulator>>> *tmp_sims_cp;
  bool dummy_circuit_cp{false};
  size_t result_len_cp{0};
  size_t encoder_g_len_cp{0};
  size_t ansatz_g_len_cp{0};
};

void ComputerForwardBackward(const std::shared_ptr<ComputeParam> &input_params, size_t start, size_t end, size_t id) {
  MS_EXCEPTION_IF_NULL(input_params);
  MS_EXCEPTION_IF_NULL(input_params->encoder_data_cp);
  MS_EXCEPTION_IF_NULL(input_params->ansatz_data_cp);
  MS_EXCEPTION_IF_NULL(input_params->output_cp);
  MS_EXCEPTION_IF_NULL(input_params->gradient_encoder_cp);
  MS_EXCEPTION_IF_NULL(input_params->gradient_ansatz_cp);
  auto encoder_data = input_params->encoder_data_cp;
  auto ansatz_data = input_params->ansatz_data_cp;
  auto output = input_params->output_cp;
  auto gradient_encoder = input_params->gradient_encoder_cp;
  auto gradient_ansatz = input_params->gradient_ansatz_cp;
  auto circ = input_params->circ_cp;
  auto herm_circ = input_params->herm_circ_cp;
  auto hams = input_params->hams_cp;
  auto encoder_params_names = input_params->encoder_params_names_cp;
  auto ansatz_params_names = input_params->ansatz_params_names_cp;
  auto tmp_sims = input_params->tmp_sims_cp;
  auto dummy_circuit = input_params->dummy_circuit_cp;
  auto result_len = input_params->result_len_cp;
  auto encoder_g_len = input_params->encoder_g_len_cp;
  auto ansatz_g_len = input_params->ansatz_g_len_cp;
  MS_EXCEPTION_IF_NULL(hams);
  MS_EXCEPTION_IF_NULL(encoder_params_names);
  MS_EXCEPTION_IF_NULL(ansatz_params_names);
  MS_EXCEPTION_IF_NULL(tmp_sims);

  if (end * hams->size() > result_len || end * encoder_params_names->size() * hams->size() > encoder_g_len ||
      end * ansatz_params_names->size() * hams->size() > ansatz_g_len) {
    MS_LOG(EXCEPTION) << "pqc error input size!";
  }
  mindquantum::ParameterResolver pr;
  for (size_t i = 0; i < ansatz_params_names->size(); i++) {
    pr.SetData(ansatz_params_names->at(i), ansatz_data[i]);
  }
  for (size_t n = start; n < end; ++n) {
    for (size_t i = 0; i < encoder_params_names->size(); i++) {
      pr.SetData(encoder_params_names->at(i), encoder_data[n * encoder_params_names->size() + i]);
    }
    auto sim = tmp_sims->at(id)[3];
    sim->SetZeroState();
    sim->Evolution(*circ, pr);
    auto calc_gradient_param = std::make_shared<mindquantum::CalcGradientParam>();
    calc_gradient_param->circuit_cp = circ;
    calc_gradient_param->circuit_hermitian_cp = herm_circ;
    calc_gradient_param->hamiltonians_cp = hams;
    calc_gradient_param->paras_cp = &pr;
    calc_gradient_param->encoder_params_names_cp = encoder_params_names;
    calc_gradient_param->ansatz_params_names_cp = ansatz_params_names;
    calc_gradient_param->dummy_circuit_cp = dummy_circuit;

    auto e0_grad1_grad_2 =
      sim->CalcGradient(calc_gradient_param, *tmp_sims->at(id)[0], *tmp_sims->at(id)[1], *tmp_sims->at(id)[2]);
    auto energy = e0_grad1_grad_2[0];
    auto grad_encoder = e0_grad1_grad_2[1];
    auto grad_ansatz = e0_grad1_grad_2[2];
    if (energy.size() != hams->size() || grad_encoder.size() != encoder_params_names->size() * hams->size() ||
        grad_ansatz.size() != ansatz_params_names->size() * hams->size()) {
      MS_LOG(EXCEPTION) << "pqc error evolution or batch size!";
    }
    for (size_t poi = 0; poi < hams->size(); poi++) {
      output[n * hams->size() + poi] = energy[poi];
    }
    for (size_t poi = 0; poi < encoder_params_names->size() * hams->size(); poi++) {
      gradient_encoder[n * hams->size() * encoder_params_names->size() + poi] = grad_encoder[poi];
    }
    for (size_t poi = 0; poi < ansatz_params_names->size() * hams->size(); poi++) {
      gradient_ansatz[n * hams->size() * ansatz_params_names->size() + poi] = grad_ansatz[poi];
    }
  }
}
}  // namespace

void PQCCPUKernel::InitPQCStructure(const CNodePtr &kernel_node) {
  n_threads_user_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, mindquantum::kNThreads);
  n_qubits_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, mindquantum::kNQubits);
  encoder_params_names_ =
    AnfAlgo::GetNodeAttr<mindquantum::transformer::NamesType>(kernel_node, mindquantum::kEncoderParamsNames);
  ansatz_params_names_ =
    AnfAlgo::GetNodeAttr<mindquantum::transformer::NamesType>(kernel_node, mindquantum::kAnsatzParamsNames);
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

void PQCCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> encoder_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> ansatz_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> result_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  std::vector<size_t> encoder_g_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 1);
  std::vector<size_t> ansatz_g_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 2);

  if (encoder_shape.size() != 2 || ansatz_shape.size() != 1 || result_shape.size() != 2 ||
      encoder_g_shape.size() != 3 || ansatz_g_shape.size() != 3) {
    MS_LOG(EXCEPTION) << "pqc invalid input size";
  }
  result_len_ = result_shape[0] * result_shape[1];
  encoder_g_len_ = encoder_g_shape[0] * encoder_g_shape[1] * encoder_g_shape[2];
  ansatz_g_len_ = ansatz_g_shape[0] * ansatz_g_shape[1] * ansatz_g_shape[2];

  n_samples_ = static_cast<unsigned>(encoder_shape[0]);
  InitPQCStructure(kernel_node);

  dummy_circuit_ = !std::any_of(gate_requires_grad_.begin(), gate_requires_grad_.end(),
                                [](const mindquantum::transformer::RequireType &rr) {
                                  return std::any_of(rr.begin(), rr.end(), [](const bool &r) { return r; });
                                });

  auto circs = mindquantum::transformer::CircuitTransfor(gate_names_, gate_matrix_, gate_obj_qubits_, gate_ctrl_qubits_,
                                                         gate_params_names_, gate_coeff_, gate_requires_grad_);
  circ_ = circs[0];
  herm_circ_ = circs[1];

  hams_ = mindquantum::transformer::HamiltoniansTransfor(hams_pauli_coeff_, hams_pauli_word_, hams_pauli_qubit_);

  n_threads_user_ = std::min(n_threads_user_, common::ThreadPool::GetInstance().GetSyncRunThreadNum());
  if (n_samples_ < n_threads_user_) {
    n_threads_user_ = n_samples_;
  }
  for (size_t i = 0; i < n_threads_user_; i++) {
    tmp_sims_.push_back({});
    for (size_t j = 0; j < 4; j++) {
      auto tmp = std::make_shared<mindquantum::PQCSimulator>(1, n_qubits_);
      tmp_sims_.back().push_back(tmp);
    }
  }
}

bool PQCCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                          const std::vector<kernel::AddressPtr> & /*workspace*/,
                          const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 2 || outputs.size() != 3) {
    MS_LOG(EXCEPTION) << "pqc error input output size!";
  }
  auto encoder_data = reinterpret_cast<float *>(inputs[0]->addr);
  auto ansatz_data = reinterpret_cast<float *>(inputs[1]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  auto gradient_encoder = reinterpret_cast<float *>(outputs[1]->addr);
  auto gradient_ansatz = reinterpret_cast<float *>(outputs[2]->addr);
  MS_EXCEPTION_IF_NULL(encoder_data);
  MS_EXCEPTION_IF_NULL(ansatz_data);
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(gradient_encoder);
  MS_EXCEPTION_IF_NULL(gradient_ansatz);

  std::vector<common::Task> tasks;
  std::vector<std::shared_ptr<ComputeParam>> thread_params;
  tasks.reserve(n_threads_user_);

  size_t end = 0;
  size_t offset = n_samples_ / n_threads_user_;
  size_t left = n_samples_ % n_threads_user_;
  for (size_t i = 0; i < n_threads_user_; ++i) {
    auto params = std::make_shared<ComputeParam>();
    params->encoder_data_cp = encoder_data;
    params->ansatz_data_cp = ansatz_data;
    params->output_cp = output;
    params->gradient_encoder_cp = gradient_encoder;
    params->gradient_ansatz_cp = gradient_ansatz;
    params->circ_cp = &circ_;
    params->herm_circ_cp = &herm_circ_;
    params->hams_cp = &hams_;
    params->encoder_params_names_cp = &encoder_params_names_;
    params->ansatz_params_names_cp = &ansatz_params_names_;
    params->tmp_sims_cp = &tmp_sims_;
    params->dummy_circuit_cp = dummy_circuit_;
    params->result_len_cp = result_len_;
    params->encoder_g_len_cp = encoder_g_len_;
    params->ansatz_g_len_cp = ansatz_g_len_;
    size_t start = end;
    end = start + offset;
    if (i < left) {
      end += 1;
    }
    auto task = [&params, start, end, i]() {
      ComputerForwardBackward(params, start, end, i);
      return common::SUCCESS;
    };
    tasks.emplace_back(task);
    thread_params.emplace_back(params);
  }
  common::ThreadPool::GetInstance().SyncRun(tasks);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
