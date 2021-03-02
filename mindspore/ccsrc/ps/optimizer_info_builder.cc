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

#include "ps/optimizer_info_builder.h"
#include <vector>
#include <memory>
#include <functional>
#include "backend/kernel_compiler/cpu/ps/sparse_apply_ftrl_ps_kernel.h"

namespace mindspore {
namespace ps {
using mindspore::kernel::ps::SparseApplyFtrlPSKernel;
OptimizerInfo *OptimizerInfoBuilder::Build(const std::shared_ptr<PServerKernel> &pserver_kernel,
                                           const WeightPtr &weight, const Keys &keys, const Values &values,
                                           const Lengths &lens, const InputsShapePtr &inputs_shape, size_t worker_num,
                                           bool sharded) {
  MS_EXCEPTION_IF_NULL(pserver_kernel);
  MS_EXCEPTION_IF_NULL(inputs_shape);
  OptimizerInfo *optim_info =
    BuildInputs(weight, keys, values, lens, inputs_shape, worker_num, pserver_kernel, sharded);
  MS_EXCEPTION_IF_NULL(optim_info);
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
    MS_EXCEPTION_IF_NULL(workspace);
    workspace->addr = new float[size];
    MS_EXCEPTION_IF_NULL(workspace->addr);
    workspace->size = size;
    info->AddWorkspace(workspace);
  }
}

template <typename T>
AddressPtr OptimizerInfoBuilder::GenInputAddrPtr(const std::string &optim_type, const std::string &input_name,
                                                 void *ps_data, const Lengths &ps_lens,
                                                 const InputsShapePtr &inputs_shape) {
  MS_EXCEPTION_IF_NULL(ps_data);
  // Take note of that the data type maybe inconsistent in ps_data.
  MS_LOG(INFO) << "Get input address pointer for optimizer:" << optim_type << ", input name:" << input_name;
  AddressPtr addr_ptr = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(addr_ptr);

  if (kOptimToOriginIdx.count(optim_type) == 0 || kOptimToPSSendIdx.count(optim_type) == 0) {
    MS_LOG(EXCEPTION) << "Optimizer type " << optim_type << " in not supported.";
  }
  const OptimOriginIdx &origin_input_map = kOptimToOriginIdx.at(optim_type);
  const OptimPSSendIdx &ps_send_index_map = kOptimToPSSendIdx.at(optim_type);
  if (ps_send_index_map.count(input_name) == 0 || origin_input_map.count(input_name) == 0) {
    MS_LOG(EXCEPTION) << "Optimizer " << optim_type << " has no input for " << input_name;
  }
  size_t ps_index = ps_send_index_map.at(input_name);
  if (ps_index == INDEX_NOT_SEND) {
    MS_LOG(EXCEPTION) << "Input " << input_name << " is not supposed to be sent to PS.";
  }

  size_t addr_data_size, addr_data_offset;
  if (inputs_shape != nullptr) {
    // addr_data_size should be calculated by inputs_shape if it's passed.
    size_t origin_index = origin_input_map.at(input_name);
    EXC_IF_VEC_IDX_OOB((*inputs_shape), origin_index);
    auto shape = *((*inputs_shape)[origin_index]);
    addr_data_size = std::accumulate(shape.begin(), shape.end(), worker_num_, std::multiplies<size_t>());
  } else {
    EXC_IF_VEC_IDX_OOB(ps_lens, ps_index);
    addr_data_size = ps_lens[ps_index];
  }
  addr_data_offset = std::accumulate(ps_lens.begin(), ps_lens.begin() + ps_index, 0, std::plus<int>());

  // The size in ps_lens instead of addr_data_size is the size of real data.
  T *buffer = new T[addr_data_size];
  addr_ptr->size = ps_lens[ps_index] * sizeof(T);
  addr_ptr->addr = buffer;

  size_t dst_size = addr_ptr->size;
  size_t src_size = addr_ptr->size;
  void *dst_data = addr_ptr->addr;
  void *src_data = reinterpret_cast<T *>(ps_data) + addr_data_offset;
  MS_EXCEPTION_IF_NULL(dst_data);
  MS_EXCEPTION_IF_NULL(src_data);
  int64_t ret = memcpy_s(dst_data, dst_size, src_data, src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    delete[] buffer;
    buffer = nullptr;
    return nullptr;
  }
  return addr_ptr;
}

OptimizerInfo *MomentumOptimInfoBuilder::BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values,
                                                     const Lengths &lens, const InputsShapePtr &inputs_shape,
                                                     size_t worker_num, const std::shared_ptr<PServerKernel> &, bool) {
  AddressPtr weight_addr = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(weight_addr);
  weight_addr->addr = weight->data();
  weight_addr->size = weight->size() * sizeof(float);

  AddressPtr accumulate = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(accumulate);
  accumulate->addr = new float[weight->size()];
  MS_EXCEPTION_IF_NULL(accumulate->addr);
  accumulate->size = weight->size() * sizeof(float);
  int64_t ret = memset_s(accumulate->addr, accumulate->size, 0x00, accumulate->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memset_s error, errorno(" << ret << ")";
    delete[] reinterpret_cast<float *>(accumulate->addr);
    accumulate->addr = nullptr;
    return nullptr;
  }

  AddressPtr learning_rate = GenInputAddrPtr<float>(kApplyMomentum, "lr", const_cast<float *>(values.data()), lens);
  AddressPtr gradient = GenInputAddrPtr<float>(kApplyMomentum, "grad", const_cast<float *>(values.data()), lens);
  AddressPtr momentum = GenInputAddrPtr<float>(kApplyMomentum, "momentum", const_cast<float *>(values.data()), lens);
  return new MomentumOptimInfo(weight_addr, accumulate, learning_rate, gradient, momentum);
}

OptimizerInfo *SparseAdamOptimInfoBuilder::BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values,
                                                       const Lengths &lens, const InputsShapePtr &inputs_shape,
                                                       size_t worker_num, const std::shared_ptr<PServerKernel> &,
                                                       bool sharded) {
  AddressPtr weight_addr = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(weight_addr);
  weight_addr->addr = weight->data();
  weight_addr->size = weight->size() * sizeof(float);

  AddressPtr m = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(m);
  m->addr = new float[weight->size()];
  MS_EXCEPTION_IF_NULL(m->addr);
  m->size = weight->size() * sizeof(float);
  int64_t ret = memset_s(m->addr, m->size, 0x00, m->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memset_s error, errorno(" << ret << ")";
    delete[] reinterpret_cast<float *>(m->addr);
    m->addr = nullptr;
    return nullptr;
  }

  AddressPtr v = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(v);
  v->addr = new float[weight->size()];
  MS_EXCEPTION_IF_NULL(v->addr);
  v->size = weight->size() * sizeof(float);
  ret = memset_s(v->addr, v->size, 0x00, v->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memset_s error, errorno(" << ret << ")";
    delete[] reinterpret_cast<float *>(v->addr);
    v->addr = nullptr;
    delete[] reinterpret_cast<float *>(m->addr);
    m->addr = nullptr;
    return nullptr;
  }

  AddressPtr beta1_power = GenInputAddrPtr<float>(kSparseAdam, "beta1_power", const_cast<float *>(values.data()), lens);
  AddressPtr beta2_power = GenInputAddrPtr<float>(kSparseAdam, "beta2_power", const_cast<float *>(values.data()), lens);
  AddressPtr learning_rate = GenInputAddrPtr<float>(kSparseAdam, "lr", const_cast<float *>(values.data()), lens);
  AddressPtr beta1 = GenInputAddrPtr<float>(kSparseAdam, "beta1", const_cast<float *>(values.data()), lens);
  AddressPtr beta2 = GenInputAddrPtr<float>(kSparseAdam, "beta2", const_cast<float *>(values.data()), lens);
  AddressPtr epsilon = GenInputAddrPtr<float>(kSparseAdam, "eps", const_cast<float *>(values.data()), lens);
  AddressPtr grad = GenInputAddrPtr<float>(kSparseAdam, "grad", const_cast<float *>(values.data()), lens, inputs_shape);
  AddressPtr indices =
    GenInputAddrPtr<float>(kSparseAdam, "indices", const_cast<float *>(values.data()), lens, inputs_shape);
  return new SparseAdamOptimInfo(weight_addr, m, v, beta1_power, beta2_power, learning_rate, beta1, beta2, epsilon,
                                 grad, indices, sharded);
}

OptimizerInfo *SparseFtrlOptimInfoBuilder::BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values,
                                                       const Lengths &lens, const InputsShapePtr &inputs_shape,
                                                       size_t worker_num,
                                                       const std::shared_ptr<PServerKernel> &pserver_kernel,
                                                       bool sharded) {
  MS_EXCEPTION_IF_NULL(inputs_shape);
  AddressPtr weight_addr = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(weight_addr);
  weight_addr->addr = weight->data();
  weight_addr->size = weight->size() * sizeof(float);

  AddressPtr accum = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(accum);
  accum->addr = new float[weight->size()];
  MS_EXCEPTION_IF_NULL(accum->addr);
  accum->size = weight->size() * sizeof(float);
  for (size_t i = 0; i < weight->size(); i++) {
    float *tmp = reinterpret_cast<float *>(accum->addr);
    tmp[i] = std::dynamic_pointer_cast<SparseApplyFtrlPSKernel>(pserver_kernel)->init_accum();
  }

  AddressPtr linear = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(linear);
  linear->addr = new float[weight->size()];
  MS_EXCEPTION_IF_NULL(linear->addr);
  int64_t ret = memset_s(linear->addr, weight->size() * sizeof(float), 0x00, weight->size() * sizeof(float));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memset_s error, errorno(" << ret << ")";
    delete[] reinterpret_cast<float *>(linear->addr);
    linear->addr = nullptr;
    return nullptr;
  }
  linear->size = weight->size() * sizeof(float);

  AddressPtr grad = GenInputAddrPtr<float>(kSparseFtrl, "grad", const_cast<float *>(values.data()), lens, inputs_shape);
  AddressPtr indices =
    GenInputAddrPtr<float>(kSparseFtrl, "indices", const_cast<float *>(values.data()), lens, inputs_shape);
  return new SparseFtrlOptimInfo(weight_addr, accum, linear, grad, indices, sharded);
}
}  // namespace ps
}  // namespace mindspore
