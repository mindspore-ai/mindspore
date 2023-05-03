/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "kernel/graph_kernel/dynamic_akg/dynamic_akg_kernel_build.h"

#include <sys/shm.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>
#include "ir/dtype.h"
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "kernel/common_utils.h"
#include "kernel/graph_kernel/graph_kernel_json_generator.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
bool DynamicAkgKernelBuilder::SingleOpParallelBuild(const std::vector<AnfNodePtr> &anf_nodes) { return true; }
}  // namespace kernel
}  // namespace mindspore
