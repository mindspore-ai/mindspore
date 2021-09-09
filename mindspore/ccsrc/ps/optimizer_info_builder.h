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

#ifndef MINDSPORE_CCSRC_PS_OPTIMIZER_INFO_BUILDER_H_
#define MINDSPORE_CCSRC_PS_OPTIMIZER_INFO_BUILDER_H_

#include <vector>
#include <memory>
#include <string>
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/cpu/ps/pserver_kernel.h"
#include "ps/optimizer_info.h"

namespace mindspore {
namespace ps {
using mindspore::kernel::KernelMod;
using mindspore::kernel::ps::PServerKernel;
class OptimizerInfoBuilder {
 public:
  explicit OptimizerInfoBuilder(size_t worker_num) : worker_num_(worker_num) {}
  virtual ~OptimizerInfoBuilder() = default;

  OptimizerInfo *Build(const std::shared_ptr<PServerKernel> &pserver_kernel, const WeightPtr &weight, const Keys &keys,
                       const Values &values, const Lengths &lens, const InputsShapePtr &inputs_shape, size_t worker_num,
                       bool sharded);

  virtual OptimizerInfo *BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values,
                                     const Lengths &lens, const InputsShapePtr &inputs_shape, size_t worker_num,
                                     const std::shared_ptr<PServerKernel> &pserver_kernel, bool sharded) = 0;

  virtual void BuildWorkspaces(OptimizerInfo *info, const std::vector<size_t> &ws_sizes, size_t worker_num);
  virtual void BuildOutputs(OptimizerInfo *info, size_t worker_num) {}

 protected:
  template <typename T>
  AddressPtr GenInputAddrPtr(const std::string &optim_type, const std::string &input_name, void *ps_data,
                             const Lengths &lens, const InputsShapePtr &inputs_shape = nullptr);

  size_t worker_num_;
};

class MomentumOptimInfoBuilder : public OptimizerInfoBuilder {
 public:
  explicit MomentumOptimInfoBuilder(size_t worker_num) : OptimizerInfoBuilder(worker_num) {}
  ~MomentumOptimInfoBuilder() = default;
  OptimizerInfo *BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values, const Lengths &lens,
                             const InputsShapePtr &inputs_shape, size_t worker_num,
                             const std::shared_ptr<PServerKernel> &pserver_kernel, bool sharded) override;
};

class SparseAdamOptimInfoBuilder : public OptimizerInfoBuilder {
 public:
  explicit SparseAdamOptimInfoBuilder(size_t worker_num) : OptimizerInfoBuilder(worker_num) {}
  ~SparseAdamOptimInfoBuilder() = default;
  OptimizerInfo *BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values, const Lengths &lens,
                             const InputsShapePtr &inputs_shape, size_t worker_num,
                             const std::shared_ptr<PServerKernel> &pserver_kernel, bool sharded) override;
};

class SparseFtrlOptimInfoBuilder : public OptimizerInfoBuilder {
 public:
  explicit SparseFtrlOptimInfoBuilder(size_t worker_num) : OptimizerInfoBuilder(worker_num) {}
  ~SparseFtrlOptimInfoBuilder() = default;
  OptimizerInfo *BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values, const Lengths &lens,
                             const InputsShapePtr &inputs_shape, size_t worker_num,
                             const std::shared_ptr<PServerKernel> &pserver_kernel, bool sharded) override;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_OPTIMIZER_INFO_BUILDER_H_
