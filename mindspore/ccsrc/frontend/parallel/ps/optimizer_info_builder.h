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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_OPTIMIZER_INFO_BUILDER_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/cpu/ps/pserver_kernel.h"
#include "frontend/parallel/ps/optimizer_info.h"

namespace mindspore {
namespace parallel {
namespace ps {
using mindspore::kernel::KernelMod;
using mindspore::kernel::ps::PServerKernel;
class OptimizerInfoBuilder {
 public:
  OptimizerInfoBuilder() = default;
  virtual ~OptimizerInfoBuilder() = default;

  OptimizerInfo *Build(const std::shared_ptr<PServerKernel> &pserver_kernel, const WeightPtr &weight, const Keys &keys,
                       const Values &values, const Lengths &lens, const InputsShapePtr &inputs_shape,
                       size_t worker_num);

  virtual OptimizerInfo *BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values,
                                     const Lengths &lens, const InputsShapePtr &inputs_shape, size_t worker_num) = 0;

  virtual void BuildWorkspaces(OptimizerInfo *info, const std::vector<size_t> &ws_sizes, size_t worker_num);
  virtual void BuildOutputs(OptimizerInfo *info, size_t worker_num) {}
};

class MomentumOptimInfoBuilder : public OptimizerInfoBuilder {
 public:
  OptimizerInfo *BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values, const Lengths &lens,
                             const InputsShapePtr &inputs_shape, size_t worker_num) override;
};

class SparseAdamOptimInfoBuilder : public OptimizerInfoBuilder {
 public:
  OptimizerInfo *BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values, const Lengths &lens,
                             const InputsShapePtr &inputs_shpae, size_t worker_num) override;
};

class SparseFtrlOptimInfoBuilder : public OptimizerInfoBuilder {
 public:
  OptimizerInfo *BuildInputs(const WeightPtr &weight, const Keys &keys, const Values &values, const Lengths &lens,
                             const InputsShapePtr &inputs_shpae, size_t worker_num) override;
};
}  // namespace ps
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_OPTIMIZER_INFO_BUILDER_H_
