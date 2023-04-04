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

#include "src/extendrt/delegate/tensorrt/distribution/distribution_collective.h"

namespace mindspore::lite {
DistributionCollective::DistributionCollective() {}

DistributionCollective &DistributionCollective::instance() {
  static DistributionCollective instance;
  return instance;
}

int DistributionCollective::ReduceScatterWrapper(const void *input_addr, void *output_addr, size_t count,
                                                 nvinfer1::DataType data_type, ReduceMode reduce_type,
                                                 cudaStream_t stream, const std::string &group) {
  return RET_OK;
}

int DistributionCollective::AllReduceWrapper(const void *input_addr, void *output_addr, size_t count,
                                             nvinfer1::DataType data_type, ReduceMode reduce_type, cudaStream_t stream,
                                             const std::string &group) {
  return RET_OK;
}

int DistributionCollective::AllGatherWrapper(const void *input_addr, void *output_addr, size_t count,
                                             nvinfer1::DataType data_type, cudaStream_t stream,
                                             const std::string &group_name) {
  return RET_OK;
}
}  // namespace mindspore::lite
