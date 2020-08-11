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

#include "frontend/parallel/ps/optimizer_info.h"
#include <memory>

namespace mindspore {
namespace parallel {
namespace ps {
void OptimizerInfo::AddWorkspace(const AddressPtr &workspace) { workspaces_.push_back(workspace); }

const std::vector<AddressPtr> &OptimizerInfo::inputs() { return inputs_; }

const std::vector<AddressPtr> &OptimizerInfo::workspaces() { return workspaces_; }

const std::vector<AddressPtr> &OptimizerInfo::outputs() { return outputs_; }

bool OptimizerInfo::IsSparse() const { return false; }

size_t OptimizerInfo::grad_index() { return 0; }

size_t OptimizerInfo::indices_index() { return 0; }

void OptimizerInfo::UpdateWeight(const WeightPtr &weight) {
  AddressPtr weight_addr = std::make_shared<kernel::Address>();
  weight_addr->addr = weight->data();
  weight_addr->size = weight->size();
  inputs_[0] = weight_addr;
}

void DenseOptimInfo::Accumulate(const Values &values, const Lengths &lengths) {
  float *accum_grad_data = reinterpret_cast<float *>(gradient()->addr);
  size_t size = gradient()->size / sizeof(float);
  size_t grad_index = this->grad_index();
  size_t grad_offset = 0;
  for (size_t i = 0; i < grad_index; i++) {
    grad_offset += lengths[i];
  }
  float *grad_data = values.data() + grad_offset;
  CHECK_EQ(size, static_cast<size_t>(lengths[grad_index]));

  for (size_t i = 0; i < size; i++) {
    accum_grad_data[i] += grad_data[i];
  }
}

void DenseOptimInfo::ComputeMean(size_t n) {
  if (n > 1) {
    float *accum_grad_data = reinterpret_cast<float *>(gradient()->addr);
    size_t size = gradient()->size / sizeof(float);
    for (size_t i = 0; i < size; i++) {
      accum_grad_data[i] /= n;
    }
  }
}

void DenseOptimInfo::Reset() { memset_s(gradient()->addr, gradient()->size, 0x00, gradient()->size); }

void SparseOptimInfo::Accumulate(const Values &values, const Lengths &lengths) {
  // Append grad data to the end
  float *accum_grad_data = reinterpret_cast<float *>(gradient()->addr);

  size_t grad_index = this->grad_index();
  size_t grad_offset = 0;
  for (size_t i = 0; i < grad_index; i++) {
    grad_offset += lengths[i];
  }
  float *incr_grad_data = values.data() + grad_offset;
  size_t incr_grad_size = lengths[grad_index] * sizeof(float);

  auto ret = memcpy_s(accum_grad_data + grads_offset_, incr_grad_size, incr_grad_data, incr_grad_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
  grads_offset_ += incr_grad_size;
  gradient()->size += incr_grad_size;

  // Append indice data to the end
  int *accum_indices_data = reinterpret_cast<int *>(indices()->addr);

  size_t indices_index = this->indices_index();
  size_t indice_offset = 0;
  for (size_t i = 0; i < indices_index; i++) {
    indice_offset += lengths[i];
  }
  int *incr_indice_data = reinterpret_cast<int *>(values.data() + indice_offset);
  size_t incr_indice_size = lengths[indices_index] * sizeof(float);

  auto ret2 = memcpy_s(accum_indices_data + indices_offset_, incr_indice_size, incr_indice_data, incr_indice_size);
  if (ret2 != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret2 << ")";
  }
  indices_offset_ += incr_indice_size;
  indices()->size += incr_indice_size;
}

void SparseOptimInfo::Reset() {
  auto &gradient = this->gradient();
  gradient->size = 0;
  auto &indices = this->indices();
  indices->size = 0;
  grads_offset_ = 0;
  indices_offset_ = 0;
}

MomentumOptimInfo::MomentumOptimInfo(const AddressPtr &weight, const AddressPtr &accumulate,
                                     const AddressPtr &learning_rate, const AddressPtr &gradient,
                                     const AddressPtr &momentum) {
  inputs_.push_back(weight);
  inputs_.push_back(accumulate);
  inputs_.push_back(learning_rate);
  inputs_.push_back(gradient);
  inputs_.push_back(momentum);
}

void MomentumOptimInfo::Update(const Values &values, const Lengths &lens) {
  size_t lr_offset = 0;
  float *lr = values.data() + lr_offset;
  auto ret = memcpy_s(inputs_[2]->addr, sizeof(float), lr, sizeof(float));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
}

const AddressPtr &MomentumOptimInfo::gradient() { return inputs_[3]; }

const AddressPtr &MomentumOptimInfo::indices() { return inputs_[3]; }

size_t MomentumOptimInfo::grad_index() { return 1; }

SparseAdamOptimInfo::SparseAdamOptimInfo(const AddressPtr &weight, const AddressPtr &m, const AddressPtr &v,
                                         const AddressPtr &beta1_power, const AddressPtr &beta2_power,
                                         const AddressPtr &learning_rate, const AddressPtr &beta1,
                                         const AddressPtr &beta2, const AddressPtr &epsilon, const AddressPtr &grad,
                                         const AddressPtr &indices) {
  inputs_.push_back(weight);
  inputs_.push_back(m);
  inputs_.push_back(v);
  inputs_.push_back(beta1_power);
  inputs_.push_back(beta2_power);
  inputs_.push_back(learning_rate);
  inputs_.push_back(beta1);
  inputs_.push_back(beta2);
  inputs_.push_back(epsilon);
  inputs_.push_back(grad);
  inputs_.push_back(indices);
  grads_offset_ = 0;
  indices_offset_ = 0;
}

void SparseAdamOptimInfo::Update(const Values &values, const Lengths &lens) {
  void *data_ptr = values.data();
  AddressPtr beta1_power = inputs_[3];
  size_t size = values.size() * sizeof(float);
  auto ret = memcpy_s(beta1_power->addr, size, data_ptr, size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
  }
}

const AddressPtr &SparseAdamOptimInfo::gradient() { return inputs_[9]; }

const AddressPtr &SparseAdamOptimInfo::indices() { return inputs_[10]; }

bool SparseAdamOptimInfo::IsSparse() const { return true; }

size_t SparseAdamOptimInfo::grad_index() { return 6; }

size_t SparseAdamOptimInfo::indices_index() { return 7; }

SparseFtrlOptimInfo::SparseFtrlOptimInfo(const AddressPtr &weight, const AddressPtr &accum, const AddressPtr &linear,
                                         const AddressPtr &grad, const AddressPtr &indices) {
  inputs_.push_back(weight);
  inputs_.push_back(accum);
  inputs_.push_back(linear);
  inputs_.push_back(grad);
  inputs_.push_back(indices);
  grads_offset_ = 0;
  indices_offset_ = 0;
}

const AddressPtr &SparseFtrlOptimInfo::gradient() { return inputs_[3]; }

const AddressPtr &SparseFtrlOptimInfo::indices() { return inputs_[4]; }

bool SparseFtrlOptimInfo::IsSparse() const { return true; }

size_t SparseFtrlOptimInfo::grad_index() { return 0; }

size_t SparseFtrlOptimInfo::indices_index() { return 1; }
}  // namespace ps
}  // namespace parallel
}  // namespace mindspore
