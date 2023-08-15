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

#include "runtime/graph_scheduler/actor/any_type_kernel_actor.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace runtime {
namespace {}  // namespace
AnyTypeKernelActor::AnyTypeKernelActor(const std::string &name, const KernelGraphPtr &graph,
                                       const DeviceContext *device_context, const AID &memory_manager_aid,
                                       const AID *debug_aid, const AID *recorder_aid, KernelTransformType type)
    : SuperKernelActor(name, graph, device_context, memory_manager_aid, debug_aid, recorder_aid, type) {}

void AnyTypeKernelActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {}

void AnyTypeKernelActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {}

bool AnyTypeKernelActor::CheckGraphOutputRunningCondition(const OpContext<DeviceTensor> *context) { return false; }

void AnyTypeKernelActor::UpdataDynamicShapeParameter(OpContext<DeviceTensor> *const context) {}

void AnyTypeKernelActor::RunForGraphInput(OpContext<DeviceTensor> *const context) {}

void AnyTypeKernelActor::EraseGraphOutput(OpContext<DeviceTensor> *const context) {}

void AnyTypeKernelActor::RunForGraphOutput(OpContext<DeviceTensor> *const context) {}

void AnyTypeKernelActor::Init() {}

void AnyTypeKernelActor::FetchGraphOutput(OpContext<DeviceTensor> *const context) {}

void AnyTypeKernelActor::UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                                          const AnfNodePtr &output_node, OpContext<DeviceTensor> *const context) {}

void AnyTypeKernelActor::SendOutput(OpContext<DeviceTensor> *const context) {}
}  // namespace runtime
}  // namespace mindspore
