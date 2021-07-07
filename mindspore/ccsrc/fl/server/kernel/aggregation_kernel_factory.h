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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_AGGREGATION_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_AGGREGATION_KERNEL_FACTORY_H_

#include <memory>
#include <string>
#include <utility>
#include "fl/server/kernel/kernel_factory.h"
#include "fl/server/kernel/aggregation_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
using AggregationKernelCreator = std::function<std::shared_ptr<AggregationKernel>()>;
class AggregationKernelFactory : public KernelFactory<std::shared_ptr<AggregationKernel>, AggregationKernelCreator> {
 public:
  static AggregationKernelFactory &GetInstance() {
    static AggregationKernelFactory instance;
    return instance;
  }

 private:
  AggregationKernelFactory() = default;
  ~AggregationKernelFactory() override = default;
  AggregationKernelFactory(const AggregationKernelFactory &) = delete;
  AggregationKernelFactory &operator=(const AggregationKernelFactory &) = delete;

  // Judge whether the server aggregation kernel can be created according to registered ParamsInfo.
  bool Matched(const ParamsInfo &params_info, const CNodePtr &kernel_node) override;
};

class AggregationKernelRegister {
 public:
  AggregationKernelRegister(const std::string &name, const ParamsInfo &params_info,
                            AggregationKernelCreator &&creator) {
    AggregationKernelFactory::GetInstance().Register(name, params_info, std::move(creator));
  }
  ~AggregationKernelRegister() = default;
};

// Register aggregation kernel with one template type T.
#define REG_AGGREGATION_KERNEL(NAME, PARAMS_INFO, CLASS, T)                                                 \
  static_assert(std::is_base_of<AggregationKernel, CLASS<T>>::value, " must be base of AggregationKernel"); \
  static const AggregationKernelRegister g_##NAME##_##T##_aggregation_kernel_reg(                           \
    #NAME, PARAMS_INFO, []() { return std::make_shared<CLASS<T>>(); });

// Register aggregation kernel with two template types: T and S.
#define REG_AGGREGATION_KERNEL_TWO(NAME, PARAMS_INFO, CLASS, T, S)                                             \
  static_assert(std::is_base_of<AggregationKernel, CLASS<T, S>>::value, " must be base of AggregationKernel"); \
  static const AggregationKernelRegister g_##NAME##_##T##_##S##_aggregation_kernel_reg(                        \
    #NAME, PARAMS_INFO, []() { return std::make_shared<CLASS<T, S>>(); });
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_AGGREGATION_KERNEL_FACTORY_H_
