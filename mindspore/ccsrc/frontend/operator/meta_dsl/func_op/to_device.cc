/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/to_device.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore::prim {
TypeId GetNodeTypeId(const AnfNodePtr &node) {
  const auto &node_abs = node->abstract();
  MS_EXCEPTION_IF_NULL(node_abs);
  auto node_abs_type = node_abs->BuildType();
  MS_EXCEPTION_IF_NULL(node_abs_type);
  auto node_type = node_abs_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(node_type);
  return node_type->element()->type_id();
}

PrimitivePtr GetCopyPrimitive(Device device) {
  static std::map<Device, PrimitivePtr> copy_prim_map = {{DEVICE_ASCEND, Prim(CopyToDevice)},
                                                         {DEVICE_NPU_LOWER, Prim(CopyToDevice)},
                                                         {DEVICE_CPU, Prim(CopyToHost)},
                                                         {DEVICE_CPU_LOWER, Prim(CopyToHost)}};
  auto iter = copy_prim_map.find(device);
  if (iter == copy_prim_map.end()) {
    MS_LOG(EXCEPTION) << "Not support to device for " << device;
  }
  return iter->second;
}

BeginFunction(ToDevice, x, device, dtype, non_blocking, copy) {
  NodePtr current = x;
  const auto &dtype_abs = dtype->abstract();
  MS_EXCEPTION_IF_NULL(dtype_abs);
  auto dtype_value = dtype_abs->BuildValue();
  if (dtype_value->isa<Int64Imm>() && static_cast<TypeId>(GetValue<int64_t>(dtype_value)) != GetNodeTypeId(x)) {
    MS_LOG(DEBUG) << "Insert Cast for dtype_value " << dtype_value->ToString();
    current = Call(Prim(Cast), current, dtype);
  } else {
    MS_LOG(DEBUG) << "No need insert cast for primitive " << prim().get() << " " << prim()->ToString();
  }

  const auto &device_abs = device->abstract();
  MS_EXCEPTION_IF_NULL(device_abs);
  auto device_value = device_abs->BuildValue();
  if (!device_value->isa<Int64Imm>()) {
    MS_LOG(DEBUG) << "Invalid device input for primitive " << prim().get() << " " << prim()->ToString();
    Return(current);
    return;
  }
  const auto &copy_prim = GetCopyPrimitive(static_cast<Device>(GetValue<int64_t>(device_value)));
  MS_LOG(DEBUG) << "Insert " << copy_prim->name() << " for primitive " << prim().get() << " " << prim()->ToString();
  const auto &non_blocking_abs = non_blocking->abstract();
  MS_EXCEPTION_IF_NULL(non_blocking_abs);
  bool non_blocking_value = GetValue<bool>(non_blocking_abs->BuildValue());
  current = Call(copy_prim, current, Value(!non_blocking_value));
  Return(current);
}
EndFunction(ToDevice)
}  // namespace mindspore::prim
