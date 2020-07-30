/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ps/optimizer_info_builder.h"
#include <functional>
#include <vector>
#include <memory>

namespace mindspore {
namespace parallel {
namespace ps {
OptimizerInfo *OptimizerInfoBuilder::Build(const std::shared_ptr<PServerKernel> &pserver_kernel,
                                           const WeightPtr &weight, const Keys &keys, const Values &values,
                                           const Lengths &lens, const InputsShapePtr &inputs_shape, size_t worker_num) {
  OptimizerInfo *optim_info = BuildInputs(weight, keys, values, lens, inputs_shape, worker_num);
  std::vector<size_t> ws_sizes = pserver_kernel->workspace_sizes();
  BuildWorkspaces(optim_info, ws_sizes, worker_num);
  BuildOutputs(optim_info, worker_num);
  return optim_info;
}

void OptimizerInfoBuilder::BuildWorkspaces(OptimizerInfo *info, const std::vector<size_t> &ws_sizes,
                                           size_t worker_num) {
  for (size_t i = 0; i < ws_sizes.size(); i++) {
    size_t size = ws_sizes[i];
    AddressPtr workspace = std::make_shared<kernel::Address>();
    workspace->addr = new float[size];
    workspace->size = size;
    info->AddWorkspace(workspace);
  }
}

OptimizerInfo *MomentumOptimInfoBuilder::BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values,
                                                     const Lengths &lens, const InputsShapePtr &inputs_shape,
                                                     size_t worker_num) {
  AddressPtr weight_addr = std::make_shared<kernel::Address>();
  weight_addr->addr = weight->data();
  weight_addr->size = weight->size() * sizeof(float);
  void *data_ptr = values.data();
  void *copy_data_ptr = new float[values.size()];
  auto ret = memcpy_s(copy_data_ptr, values.size() * sizeof(float), data_ptr, values.size() * sizeof(float));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
  AddressPtr accumulate = std::make_shared<kernel::Address>();
  accumulate->addr = new float[weight->size()];
  accumulate->size = weight->size() * sizeof(float);
  memset_s(accumulate->addr, accumulate->size, 0x00, accumulate->size);
  AddressPtr learning_rate = std::make_shared<kernel::Address>();
  learning_rate->addr = copy_data_ptr;
  learning_rate->size = lens[0] * sizeof(float);
  AddressPtr gradient = std::make_shared<kernel::Address>();
  gradient->addr = reinterpret_cast<float *>(learning_rate->addr) + lens[0];
  gradient->size = lens[1] * sizeof(float);
  AddressPtr momentum = std::make_shared<kernel::Address>();
  momentum->addr = reinterpret_cast<float *>(gradient->addr) + lens[1];
  momentum->size = lens[2] * sizeof(float);

  return new MomentumOptimInfo(weight_addr, accumulate, learning_rate, gradient, momentum);
}

OptimizerInfo *SparseAdamOptimInfoBuilder::BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values,
                                                       const Lengths &lens, const InputsShapePtr &inputs_shape,
                                                       size_t worker_num) {
  AddressPtr weight_addr = std::make_shared<kernel::Address>();
  weight_addr->addr = weight->data();
  weight_addr->size = weight->size() * sizeof(float);
  AddressPtr m = std::make_shared<kernel::Address>();
  m->addr = new float[weight->size()];
  m->size = weight->size() * sizeof(float);
  int ret = memset_s(m->addr, m->size, 0x00, m->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
  AddressPtr v = std::make_shared<kernel::Address>();
  v->addr = new float[weight->size()];
  v->size = weight->size() * sizeof(float);
  ret = memset_s(v->addr, v->size, 0x00, v->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }

  void *data_ptr = values.data();
  void *copy_data_ptr = new float[values.size()];
  ret = memcpy_s(copy_data_ptr, values.size() * sizeof(float), data_ptr, values.size() * sizeof(float));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }

  AddressPtr beta1_power = std::make_shared<kernel::Address>();
  beta1_power->addr = copy_data_ptr;
  beta1_power->size = lens[0] * sizeof(float);
  AddressPtr beta2_power = std::make_shared<kernel::Address>();
  beta2_power->addr = reinterpret_cast<float *>(beta1_power->addr) + lens[0];
  beta2_power->size = lens[1] * sizeof(float);

  AddressPtr learning_rate = std::make_shared<kernel::Address>();
  learning_rate->addr = reinterpret_cast<float *>(beta2_power->addr) + lens[1];
  learning_rate->size = lens[2] * sizeof(float);

  AddressPtr beta1 = std::make_shared<kernel::Address>();
  beta1->addr = reinterpret_cast<float *>(learning_rate->addr) + lens[2];
  beta1->size = lens[3] * sizeof(float);

  AddressPtr beta2 = std::make_shared<kernel::Address>();
  beta2->addr = reinterpret_cast<float *>(beta1->addr) + lens[3];
  beta2->size = lens[4] * sizeof(float);

  AddressPtr epsilon = std::make_shared<kernel::Address>();
  epsilon->addr = reinterpret_cast<float *>(beta2->addr) + lens[4];
  epsilon->size = lens[5] * sizeof(float);

  const std::shared_ptr<std::vector<size_t>> &grad_shape = (*inputs_shape)[9];
  size_t total_grad_size =
    std::accumulate((*grad_shape).begin(), (*grad_shape).end(), sizeof(float), std::multiplies<size_t>());
  AddressPtr grad = std::make_shared<kernel::Address>();
  grad->addr = new float[total_grad_size * worker_num];
  ret = memcpy_s(grad->addr, lens[6] * sizeof(float), reinterpret_cast<float *>(epsilon->addr) + lens[5],
                 lens[6] * sizeof(float));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
  grad->size = lens[6] * sizeof(float);

  const std::shared_ptr<std::vector<size_t>> &indices_shape = (*inputs_shape)[10];
  size_t total_indice_size =
    std::accumulate((*indices_shape).begin(), (*indices_shape).end(), sizeof(float), std::multiplies<size_t>());
  AddressPtr indices = std::make_shared<kernel::Address>();
  indices->addr = new float[total_indice_size * worker_num];
  ret = memcpy_s(indices->addr, lens[7] * sizeof(float), reinterpret_cast<float *>(epsilon->addr) + lens[5] + lens[6],
                 lens[7] * sizeof(float));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
  indices->size = lens[7] * sizeof(int);

  return new SparseAdamOptimInfo(weight_addr, m, v, beta1_power, beta2_power, learning_rate, beta1, beta2, epsilon,
                                 grad, indices);
}

OptimizerInfo *SparseFtrlOptimInfoBuilder::BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values,
                                                       const Lengths &lens, const InputsShapePtr &inputs_shape,
                                                       size_t worker_num) {
  AddressPtr weight_addr = std::make_shared<kernel::Address>();
  weight_addr->addr = weight->data();
  weight_addr->size = weight->size() * sizeof(float);
  AddressPtr accum = std::make_shared<kernel::Address>();
  accum->addr = new float[weight->size()];
  accum->size = weight->size() * sizeof(float);
  for (size_t i = 0; i < weight->size(); i++) {
    float *tmp = reinterpret_cast<float *>(accum->addr);
    tmp[i] = 1.0;
  }
  AddressPtr linear = std::make_shared<kernel::Address>();
  linear->addr = new float[weight->size()];
  int ret = memset_s(linear->addr, weight->size() * sizeof(float), 0x00, weight->size() * sizeof(float));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memset_s error, errorno(" << ret << ")";
  }
  linear->size = weight->size() * sizeof(float);

  const std::shared_ptr<std::vector<size_t>> &grad_shape = (*inputs_shape)[3];
  size_t total_grad_size = std::accumulate((*grad_shape).begin(), (*grad_shape).end(), 1, std::multiplies<size_t>());
  AddressPtr grad = std::make_shared<kernel::Address>();
  grad->addr = new float[total_grad_size * worker_num];
  ret = memcpy_s(grad->addr, lens[0] * sizeof(float), values.data(), lens[0] * sizeof(float));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
  grad->size = lens[0] * sizeof(float);

  const std::shared_ptr<std::vector<size_t>> &indices_shape = (*inputs_shape)[4];
  size_t total_indice_size =
    std::accumulate((*indices_shape).begin(), (*indices_shape).end(), 1, std::multiplies<size_t>());
  AddressPtr indices = std::make_shared<kernel::Address>();
  indices->addr = new float[total_indice_size * worker_num];
  ret = memcpy_s(indices->addr, lens[1] * sizeof(float), reinterpret_cast<float *>(values.data()) + lens[0],
                 lens[1] * sizeof(float));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
  indices->size = lens[1] * sizeof(int);

  return new SparseFtrlOptimInfo(weight_addr, accum, linear, grad, indices);
}
}  // namespace ps
}  // namespace parallel
}  // namespace mindspore
