/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_AICPU_AICPU_UTIL_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_AICPU_AICPU_UTIL_H_

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
constexpr auto kInitDataSetQueue = "InitDataSetQueue";
constexpr auto kInitData = "InitData";
constexpr auto kGetNext = "GetNext";
constexpr auto kPrint = "Print";
constexpr auto kPack = "Pack";

constexpr auto kOutputTypes = "output_types";
constexpr auto kOutputShapes = "output_shapes";
constexpr auto kChannelName = "channel_name";
constexpr auto kSharedName = "shared_name";
constexpr auto kShapes = "shapes";
constexpr auto kTypes = "types";
constexpr auto kQueueName = "queue_name";

constexpr auto kSeed = "Seed0";
constexpr auto kSeed2 = "Seed1";

struct AicpuParamHead {
  uint32_t length;         // Total length: include cunstom message
  uint32_t ioAddrNum;      // Input and output address number
  uint32_t extInfoLength;  // extInfo struct Length
  uint64_t extInfoAddr;    // extInfo address
} __attribute__((packed));

class AicpuOpUtil {
 public:
  static int MsTypeToProtoType(TypeId ms_type);

 private:
  // kernel id
  static uint64_t KernelId_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_AICPU_AICPU_UTIL_H_
