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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/inplace_to_device.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore::prim {
PrimitivePtr GetInplaceCopyPrimitive(Device device) {
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

BeginFunction(InplaceToDevice, x, device, non_blocking) {
  const auto &device_abs = device->abstract();
  MS_EXCEPTION_IF_NULL(device_abs);
  auto device_value = device_abs->BuildValue();
  if (!device_value->isa<Int64Imm>()) {
    MS_LOG(DEBUG) << "Invalid device input for primitive " << prim().get() << " " << prim()->ToString();
    Return(x);
    return;
  }
  const auto &copy_prim = GetInplaceCopyPrimitive(static_cast<Device>(GetValue<int64_t>(device_value)));
  MS_LOG(DEBUG) << "Insert " << copy_prim->name() << " for primitive " << prim().get() << " " << prim()->ToString();
  const auto &non_blocking_abs = non_blocking->abstract();
  MS_EXCEPTION_IF_NULL(non_blocking_abs);
  bool non_blocking_value = GetValue<bool>(non_blocking_abs->BuildValue());
  auto copy_node = Call(copy_prim, x, Value(!non_blocking_value));
  auto free_node = Call(Prim(Free), x, Value(!non_blocking_value));
  auto set_data_node = Call(Prim(SetData), x, copy_node);
  auto depend_node = Call(Prim(Depend), set_data_node, free_node);
  Return(depend_node);
}
EndFunction(InplaceToDevice)
}  // namespace mindspore::prim
