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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_DISTRIBUTION_DISTRIBUTION_COLLECTIVE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_DISTRIBUTION_DISTRIBUTION_COLLECTIVE_H_

#include <string>
#include "NvInfer.h"
#include "mindapi/base/types.h"
#include "src/extendrt/delegate/tensorrt/distribution/distribution_base.h"

namespace mindspore::lite {
class DistributionCollective {
 public:
  DistributionCollective(DistributionCollective const &) = delete;

  DistributionCollective &operator=(const DistributionCollective &) = delete;

  static DistributionCollective &instance();

  int ReduceScatterWrapper(const void *input_addr, void *output_addr, size_t count, nvinfer1::DataType data_type,
                           ReduceMode reduce_type, cudaStream_t stream, const std::string &group);

  int AllGatherWrapper(const void *input_addr, void *output_addr, size_t count, nvinfer1::DataType data_type,
                       cudaStream_t stream, const std::string &group_name);

 private:
  DistributionCollective();

  ~DistributionCollective() = default;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_DISTRIBUTION_DISTRIBUTION_COLLECTIVE_H_
