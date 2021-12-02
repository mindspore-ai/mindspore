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

#ifndef MINDSPORE_CCSRC_PS_CORE_CLUSTER_METADATA_H_
#define MINDSPORE_CCSRC_PS_CORE_CLUSTER_METADATA_H_

#include <string>
#include <iostream>
#include <memory>
#include <utility>

#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace core {
/*
 * The metadata information of the cluster, stored in the scheduler, is generally used for scale out and scale in.
 */
struct ClusterMetadata {
  ClusterMetadata(const uint32_t &worker, const uint32_t &server) : worker_num(worker), server_num(server) {}
  uint32_t worker_num;
  uint32_t server_num;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_CLUSTER_METADATA_H_
