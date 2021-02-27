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

#ifndef MINDSPORE_CCSRC_PS_OPTIMIZER_INFO_H_
#define MINDSPORE_CCSRC_PS_OPTIMIZER_INFO_H_

#include <vector>
#include <string>
#include "backend/kernel_compiler/kernel.h"
#include "ps/constants.h"

namespace mindspore {
namespace ps {
using mindspore::kernel::AddressPtr;
class OptimizerInfo {
 public:
  OptimizerInfo() = default;
  virtual ~OptimizerInfo() = default;

  virtual void Update(const Values &values, const Lengths &lengths) {}
  virtual void Accumulate(const Values &values, const Lengths &lengths) = 0;
  virtual void ComputeMean(const std::vector<std::vector<size_t>> &shapes, size_t n, size_t server_num,
                           size_t rank_id) {}
  virtual void Reset() {}
  void AddWorkspace(const AddressPtr &workspace);

  virtual const AddressPtr &gradient() = 0;
  virtual const AddressPtr &indices() = 0;
  virtual const size_t indice_size() const;
  const std::vector<AddressPtr> &inputs();
  const std::vector<AddressPtr> &workspaces();
  const std::vector<AddressPtr> &outputs();

  virtual bool IsSparse() const;
  virtual size_t grad_index();
  virtual size_t indices_index();

 protected:
  template <typename T>
  void UpdateOptimInputValue(const std::string &optim_type, const std::string &input_name, void *data,
                             const Lengths &lens);
  std::vector<AddressPtr> inputs_;
  std::vector<AddressPtr> workspaces_;
  std::vector<AddressPtr> outputs_;
};

class DenseOptimInfo : public OptimizerInfo {
 public:
  DenseOptimInfo() = default;
  ~DenseOptimInfo() override = default;

  void Accumulate(const Values &values, const Lengths &lens) override;
  void ComputeMean(const std::vector<std::vector<size_t>> &shapes, size_t n, size_t server_num,
                   size_t rank_id) override;
  void Reset() override;
};

class SparseOptimInfo : public OptimizerInfo {
 public:
  SparseOptimInfo() = default;
  ~SparseOptimInfo() override = default;

  void Accumulate(const Values &values, const Lengths &lens) override;
  void ComputeMean(const std::vector<std::vector<size_t>> &shapes, size_t n, size_t server_num,
                   size_t rank_id) override;
  void Reset() override;
  const size_t indice_size() const override;

 protected:
  size_t grads_offset_{0};
  size_t indices_offset_{0};
  bool sharded_{true};
};

class MomentumOptimInfo : public DenseOptimInfo {
 public:
  MomentumOptimInfo(const AddressPtr &weight, const AddressPtr &accumulate, const AddressPtr &learning_rate,
                    const AddressPtr &gradient, const AddressPtr &momentum);
  ~MomentumOptimInfo() override = default;

  void Update(const Values &values, const Lengths &lens) override;
  const AddressPtr &gradient();
  const AddressPtr &indices();
  size_t grad_index() override;
};

class SparseAdamOptimInfo : public SparseOptimInfo {
 public:
  SparseAdamOptimInfo(const AddressPtr &weight, const AddressPtr &m, const AddressPtr &v, const AddressPtr &beta1_power,
                      const AddressPtr &beta2_power, const AddressPtr &learning_rate, const AddressPtr &beta1,
                      const AddressPtr &beta2, const AddressPtr &epsilon, const AddressPtr &grad,
                      const AddressPtr &indices, bool sharded);
  ~SparseAdamOptimInfo() override = default;

  void Update(const Values &values, const Lengths &lens) override;
  const AddressPtr &gradient();
  const AddressPtr &indices();
  bool IsSparse() const override;
  size_t grad_index() override;
  size_t indices_index() override;
};

class SparseFtrlOptimInfo : public SparseOptimInfo {
 public:
  SparseFtrlOptimInfo(const AddressPtr &weight, const AddressPtr &accum, const AddressPtr &linear,
                      const AddressPtr &grad, const AddressPtr &indices, bool sharded);
  ~SparseFtrlOptimInfo() override = default;

  const AddressPtr &gradient();
  const AddressPtr &indices();
  bool IsSparse() const override;
  size_t grad_index() override;
  size_t indices_index() override;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_OPTIMIZER_INFO_H_
