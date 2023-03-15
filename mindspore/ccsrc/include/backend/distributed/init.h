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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_INIT_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_INIT_H_

#include "include/backend/distributed/collective/collective_manager.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/cluster/cluster_context.h"
#else
#include "include/backend/distributed/cluster/dummy_cluster_context.h"
#endif
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
// The static methods of MindSpore distributed execution. They can be exported by Pybind.

// Initialize and finalize distributed execution.
BACKEND_EXPORT bool Initialize();
BACKEND_EXPORT bool Finalize();

// Initialize and finalize the cluster based on MindSpore communication framework.
BACKEND_EXPORT bool InitializeCluster();
BACKEND_EXPORT bool FinalizeCluster();

// Initialize and finalize collective communication for distributed execution.
BACKEND_EXPORT bool InitializeCollective();
BACKEND_EXPORT bool FinalizeCollective();

// Set and get whether this process in cluster exits with exception.
BACKEND_EXPORT void set_cluster_exit_with_exception();
BACKEND_EXPORT bool cluster_exit_with_exception();
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_INIT_H_
