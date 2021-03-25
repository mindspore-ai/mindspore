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

#include "ps/optimizer_info.h"
#include <map>
#include <memory>
#include <string>
#include <functional>
#include "ps/util.h"

namespace mindspore {
namespace ps {
void OptimizerInfo::AddWorkspace(const AddressPtr &workspace) { workspaces_.push_back(workspace); }

const std::vector<AddressPtr> &OptimizerInfo::inputs() { return inputs_; }

const std::vector<AddressPtr> &OptimizerInfo::workspaces() { return workspaces_; }

const std::vector<AddressPtr> &OptimizerInfo::outputs() { return outputs_; }

bool OptimizerInfo::IsSparse() const { return false; }

const size_t OptimizerInfo::indice_size() const { return 0; }

size_t OptimizerInfo::grad_index() { return 0; }

size_t OptimizerInfo::indices_index() { return 0; }

template <typename T>
void OptimizerInfo::UpdateOptimInputValue(const std::string &optim_type, const std::string &input_name, void *data,
                                          const Lengths &lens) {
  if (kOptimToOriginIdx.count(optim_type) == 0 || kOptimToPSSendIdx.count(optim_type) == 0) {
    MS_LOG(EXCEPTION) << "Optimizer type " << optim_type << " in not supported.";
  }
  const OptimOriginIdx &origin_input_map = kOptimToOriginIdx.at(optim_type);
  const OptimPSSendIdx &ps_send_index_map = kOptimToPSSendIdx.at(optim_type);
  if (ps_send_index_map.count(input_name) == 0 || origin_input_map.count(input_name) == 0) {
    MS_LOG(EXCEPTION) << "Optimizer " << optim_type << " has no input for " << input_name;
  }

  size_t origin_index = origin_input_map.at(input_name);
  size_t ps_send_index = ps_send_index_map.at(input_name);
  if (ps_send_index > lens.size() || origin_index > inputs_.size()) {
    MS_LOG(EXCEPTION) << "Index is out of bound for optimizer " << optim_type << ", origin_index:" << origin_index
                      << ", ps_send_index:" << ps_send_index;
  }
  EXC_IF_VEC_IDX_OOB(lens, ps_send_index);
  size_t size = lens[ps_send_index] * sizeof(T);
  size_t offset = std::accumulate(lens.begin(), lens.begin() + ps_send_index, 0, std::plus<int>());
  AddressPtr optim_input = inputs_[origin_index];
  MS_EXCEPTION_IF_NULL(optim_input);

  void *dst_data = optim_input->addr;
  T *src_data = reinterpret_cast<T *>(data) + offset;
  MS_EXCEPTION_IF_NULL(dst_data);
  MS_EXCEPTION_IF_NULL(src_data);
  int64_t ret = memcpy_s(optim_input->addr, optim_input->size, src_data, size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }
  return;
}

void DenseOptimInfo::Accumulate(const Values &values, const Lengths &lengths) {
  MS_EXCEPTION_IF_NULL(gradient()->addr);
  float *accum_grad_data = reinterpret_cast<float *>(gradient()->addr);
  size_t size = gradient()->size / sizeof(float);
  size_t grad_index = this->grad_index();
  size_t grad_offset = 0;
  for (size_t i = 0; i < grad_index; i++) {
    grad_offset += lengths[i];
  }
  float *grad_data = const_cast<float *>(values.data()) + grad_offset;
#define google mindspore_private
  CHECK_EQ(size, static_cast<size_t>(lengths[grad_index]));
#undef google
  for (size_t i = 0; i < size; i++) {
    accum_grad_data[i] += grad_data[i];
  }
}

void DenseOptimInfo::ComputeMean(const std::vector<std::vector<size_t>> &, size_t n, size_t, size_t) {
  if (n > 1) {
    float *accum_grad_data = reinterpret_cast<float *>(gradient()->addr);
    size_t size = gradient()->size / sizeof(float);
    for (size_t i = 0; i < size; i++) {
      accum_grad_data[i] /= n;
    }
  }
}

void DenseOptimInfo::Reset() {
  MS_EXCEPTION_IF_NULL(gradient()->addr);
  int64_t ret = memset_s(gradient()->addr, gradient()->size, 0x00, gradient()->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memset_s error, errorno(" << ret << ")";
    return;
  }
}

void SparseOptimInfo::Accumulate(const Values &values, const Lengths &lengths) {
  // Append grad data to the end
  float *accum_grad_data = reinterpret_cast<float *>(gradient()->addr);
  MS_EXCEPTION_IF_NULL(accum_grad_data);

  size_t grad_index = this->grad_index();
  size_t grad_offset = 0;
  for (size_t i = 0; i < grad_index; i++) {
    grad_offset += lengths[i];
  }
  float *incr_grad_data = const_cast<float *>(values.data()) + grad_offset;
  MS_EXCEPTION_IF_NULL(incr_grad_data);

  size_t incr_grad_size = lengths[grad_index] * sizeof(float);
  size_t dst_size = incr_grad_size;
  size_t src_size = incr_grad_size;
  void *dst_data = accum_grad_data + grads_offset_;
  void *src_data = incr_grad_data;
  MS_EXCEPTION_IF_NULL(dst_data);
  MS_EXCEPTION_IF_NULL(src_data);
  int64_t ret = memcpy_s(dst_data, dst_size, src_data, src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }
  grads_offset_ += lengths[grad_index];
  gradient()->size += incr_grad_size;

  // Append indice data to the end
  int *accum_indices_data = reinterpret_cast<int *>(indices()->addr);
  MS_EXCEPTION_IF_NULL(accum_indices_data);

  size_t indices_index = this->indices_index();
  size_t indice_offset = 0;
  for (size_t i = 0; i < indices_index; i++) {
    indice_offset += lengths[i];
  }

  void *incr_indice_data_temp = const_cast<float *>(values.data()) + indice_offset;

  int *incr_indice_data = reinterpret_cast<int *>(incr_indice_data_temp);

  MS_EXCEPTION_IF_NULL(incr_indice_data);
  size_t incr_indice_size = lengths[indices_index];
  size_t incr_indice_data_size = incr_indice_size * sizeof(int);
  dst_size = incr_indice_data_size;
  src_size = incr_indice_data_size;
  dst_data = accum_indices_data + indices_offset_;
  src_data = incr_indice_data;
  MS_EXCEPTION_IF_NULL(dst_data);
  MS_EXCEPTION_IF_NULL(src_data);
  auto ret2 = memcpy_s(dst_data, dst_size, src_data, src_size);
  if (ret2 != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret2 << ")";
    return;
  }
  indices_offset_ += lengths[indices_index];
  indices()->size += incr_indice_data_size;
}

void SparseOptimInfo::ComputeMean(const std::vector<std::vector<size_t>> &shapes, size_t n, size_t server_num,
                                  size_t rank_id) {
  MS_EXCEPTION_IF_NULL(gradient());
  MS_EXCEPTION_IF_NULL(indices());
  size_t indices_size = static_cast<size_t>(indices()->size / sizeof(int));
  size_t segment_size = gradient()->size / indices()->size;

  std::vector<float> new_grad(indices_size * segment_size);
  std::vector<int> new_indices(indices_size);
  mindspore::kernel::SparseGradient<int> unique_sparse_grad({new_grad.data(), new_indices.data(), indices_size});

  if (shapes.size() < 2 || shapes[1].empty()) {
    MS_LOG(EXCEPTION) << "No input shape found";
  }
  auto input_shapes = shapes[1];
  if (input_shapes.size() == 0) {
    MS_LOG(EXCEPTION) << "Invalid input shapes";
  }
  size_t first_dim_size = input_shapes.front();
  size_t outer_dim_size = segment_size;

  if (first_dim_size == 0 || outer_dim_size == 0) {
    MS_LOG(ERROR) << "Invalid first dim size";
  }

  MS_EXCEPTION_IF_NULL(gradient()->addr);
  MS_EXCEPTION_IF_NULL(indices()->addr);
  float *grad_data = reinterpret_cast<float *>(gradient()->addr);
  int *indices_data = reinterpret_cast<int *>(indices()->addr);

  if (sharded_) {
    size_t original_row_count = input_shapes.front();
    if (original_row_count > 0) {
      size_t offset = 0;
      std::map<int64_t, int64_t> rank_dims = Util::AllRankLocalShard(original_row_count, rank_id, server_num);
      for (size_t i = 0; i < rank_id; i++) {
        if (rank_dims.count(i) == 0) {
          MS_LOG(EXCEPTION) << "No local shard number for rank " << i;
        }
        offset += rank_dims[i];
      }
      for (size_t i = 0; i < indices_size; i++) {
        indices_data[i] -= offset;
      }
    }
  }

  Util::ReduceSparseGradient(grad_data, indices_data, indices_size, segment_size, first_dim_size, outer_dim_size,
                             &unique_sparse_grad);

  int64_t reduced_grad_size = unique_sparse_grad.indices_size_ * segment_size * sizeof(float);
  MS_EXCEPTION_IF_NULL(unique_sparse_grad.value_);
  int64_t ret = memcpy_s(gradient()->addr, gradient()->size, unique_sparse_grad.value_, reduced_grad_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }

  int64_t reduced_indice_size = unique_sparse_grad.indices_size_ * sizeof(int);
  MS_EXCEPTION_IF_NULL(unique_sparse_grad.indices_);
  ret = memcpy_s(indices()->addr, indices()->size, unique_sparse_grad.indices_, reduced_indice_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }

  gradient()->size = reduced_grad_size;
  indices()->size = reduced_indice_size;

  for (size_t i = 0; i < unique_sparse_grad.indices_size_ * segment_size; i++) {
    grad_data[i] = grad_data[i] / n;
  }
}

void SparseOptimInfo::Reset() {
  gradient()->size = 0;
  indices()->size = 0;
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
  UpdateOptimInputValue<float>(kApplyMomentum, "lr", const_cast<float *>(values.data()), lens);
}

const size_t SparseOptimInfo::indice_size() const { return indices_offset_; }

const AddressPtr &MomentumOptimInfo::gradient() {
  size_t origin_grad_index = kMomentumOriginIdx.at("grad");
  EXC_IF_VEC_IDX_OOB(inputs_, origin_grad_index);
  return inputs_[origin_grad_index];
}

const AddressPtr &MomentumOptimInfo::indices() {
  size_t origin_grad_index = kMomentumOriginIdx.at("grad");
  EXC_IF_VEC_IDX_OOB(inputs_, origin_grad_index);
  return inputs_[origin_grad_index];
}

size_t MomentumOptimInfo::grad_index() {
  size_t ps_grad_index = kMomentumPSSendIdx.at("grad");
  return ps_grad_index;
}

SparseAdamOptimInfo::SparseAdamOptimInfo(const AddressPtr &weight, const AddressPtr &m, const AddressPtr &v,
                                         const AddressPtr &beta1_power, const AddressPtr &beta2_power,
                                         const AddressPtr &learning_rate, const AddressPtr &beta1,
                                         const AddressPtr &beta2, const AddressPtr &epsilon, const AddressPtr &grad,
                                         const AddressPtr &indices, bool sharded) {
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
  grads_offset_ = grad->size / sizeof(float);
  indices_offset_ = indices->size / sizeof(int);
  sharded_ = sharded;
}

void SparseAdamOptimInfo::Update(const Values &values, const Lengths &lens) {
  UpdateOptimInputValue<float>(kSparseAdam, "beta1_power", const_cast<float *>(values.data()), lens);
  UpdateOptimInputValue<float>(kSparseAdam, "beta2_power", const_cast<float *>(values.data()), lens);
  UpdateOptimInputValue<float>(kSparseAdam, "lr", const_cast<float *>(values.data()), lens);
  UpdateOptimInputValue<float>(kSparseAdam, "beta1", const_cast<float *>(values.data()), lens);
  UpdateOptimInputValue<float>(kSparseAdam, "beta2", const_cast<float *>(values.data()), lens);
  UpdateOptimInputValue<float>(kSparseAdam, "eps", const_cast<float *>(values.data()), lens);
}

const AddressPtr &SparseAdamOptimInfo::gradient() {
  size_t origin_grad_index = kSparseAdamOriginIdx.at("grad");
  EXC_IF_VEC_IDX_OOB(inputs_, origin_grad_index);
  return inputs_[origin_grad_index];
}

const AddressPtr &SparseAdamOptimInfo::indices() {
  size_t origin_indices_index = kSparseAdamOriginIdx.at("indices");
  EXC_IF_VEC_IDX_OOB(inputs_, origin_indices_index);
  return inputs_[origin_indices_index];
}

bool SparseAdamOptimInfo::IsSparse() const { return true; }

size_t SparseAdamOptimInfo::grad_index() {
  size_t ps_grad_index = kSparseAdamPSSendIdx.at("grad");
  return ps_grad_index;
}

size_t SparseAdamOptimInfo::indices_index() {
  size_t ps_indices_index = kSparseAdamPSSendIdx.at("indices");
  return ps_indices_index;
}

SparseFtrlOptimInfo::SparseFtrlOptimInfo(const AddressPtr &weight, const AddressPtr &accum, const AddressPtr &linear,
                                         const AddressPtr &grad, const AddressPtr &indices, bool sharded) {
  inputs_.push_back(weight);
  inputs_.push_back(accum);
  inputs_.push_back(linear);
  inputs_.push_back(grad);
  inputs_.push_back(indices);
  grads_offset_ = grad->size / sizeof(float);
  indices_offset_ = indices->size / sizeof(int);
  sharded_ = sharded;
}

const AddressPtr &SparseFtrlOptimInfo::gradient() {
  size_t origin_grad_index = kSparseFtrlOriginIdx.at("grad");
  EXC_IF_VEC_IDX_OOB(inputs_, origin_grad_index);
  return inputs_[origin_grad_index];
}

const AddressPtr &SparseFtrlOptimInfo::indices() {
  size_t origin_indices_index = kSparseFtrlOriginIdx.at("indices");
  EXC_IF_VEC_IDX_OOB(inputs_, origin_indices_index);
  return inputs_[origin_indices_index];
}

bool SparseFtrlOptimInfo::IsSparse() const { return true; }

size_t SparseFtrlOptimInfo::grad_index() {
  size_t ps_grad_index = kSparseFtrlPSSendIdx.at("grad");
  return ps_grad_index;
}

size_t SparseFtrlOptimInfo::indices_index() {
  size_t ps_indices_index = kSparseFtrlPSSendIdx.at("indices");
  return ps_indices_index;
}
}  // namespace ps
}  // namespace mindspore
