/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_MPMD_MODE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_MPMD_MODE_H_

#include <string>
#include <vector>
#include <memory>
#include "include/backend/distributed/mode/execution_mode.h"

namespace mindspore {
namespace distributed {
// In MPMD mode, different sub-graphs will be split to different device lists and connect to each other by P2P and
// collective communication operators. Meanwhile, data parallel, model parallel and pipeline parallel could be applied
// to these sub-graphs, which enbles advanced and flexible distributed computing algorithm, and it could be controlled
// by user through python API.

// In MPMD mode, DAG distribution could be like this:
// |------Model1------|------Model2------|------Model3------|
// |-----512 cards----|-----128 cards----|-----128 cards----|
// Model1, 2, 3 execute on different device list.

class MPMDMode : public DistributedExecutionMode {
 public:
  MPMDMode() = default;
  ~MPMDMode() override = default;
};
using MPMDModePtr = std::shared_ptr<MPMDMode>;
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_MPMD_MODE_H_
