/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_BUILD_ASCEND_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_BUILD_ASCEND_H_

#include <vector>
#include <map>
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
using CommOpInputInfo = std::map<AnfNodePtr, std::vector<size_t>>;
using CleanOpsMap = std::map<CNodePtr, std::vector<CNodePtr>>;

/**
 * @brief kernel build for ascend.
 */
bool KernelBuild(const std::vector<CNodePtr> &kernels);

/**
 * @brief preprocess of kernel build for ascend, e.g. inserting clear_zero node for max_pool, bn.
 * Must DO these changes just before kernel build, and after all of other optimizations on AnfGraph
 */
void InsertAtomicCleanOp(const KernelGraphPtr &kernel_graph);

/**
 *  @brief preprocess for mind rt
 * */
void InsertAtomicCleanOpForMindRT(const std::vector<CNodePtr> &exe_orders, CleanOpsMap *maps);

/**
 * @brief communication op input info.
 * */
CommOpInputInfo GetCommunicationOpInputInfo(const std::vector<CNodePtr> &exe_orders);

/**
 * @brief insert atomic
 * */
void InsertAtomicOps(const std::vector<CNodePtr> &exe_orders, CleanOpsMap *clean_ops);

/**
 * @brief gather all atomics
 * */
std::vector<CNodePtr> GatherAllAtomicOps(const CleanOpsMap &node_maps);

/**
 * @brief add attr for op if need insert atomic
 * */
void AddNeedInsertAtomicAttrForAllOps(const std::vector<CNodePtr> &exe_orders);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_BUILD_ASCEND_H_
