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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AICPU_EXT_INFO_HANDLE_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AICPU_EXT_INFO_HANDLE_H_

#include <string>
#include <vector>
#include <utility>
#include <memory>
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "include/common/utils/contract.h"
#include "cce/fwk_adpt_struct.h"
#include "external/graph/types.h"
#include "cce/aicpu_engine_struct.h"

namespace mindspore {
namespace device {
namespace ascend {
using AicpuShapeAndType = aicpu::FWKAdapter::ShapeAndType;
using AicpuExtInfo = aicpu::FWKAdapter::ExtInfo;
using AsyncWaitInfo = aicpu::FWKAdapter::AsyncWait;
using AicpuSessionInfo = SessionInfo;

class AicpuExtInfoHandler {
 public:
  AicpuExtInfoHandler(std::string node_name, uint32_t input_num, uint32_t output_num,
                      ::ge::UnknowShapeOpType unknown_type)
      : node_name_(std::move(node_name)),
        input_num_(input_num),
        output_num_(output_num),
        unknown_type_(unknown_type),
        ext_info_len_(0) {}

  ~AicpuExtInfoHandler() = default;

  uint8_t *GetExtInfo() const { return ext_info_.get(); }
  size_t GetExtInfoLen() const { return ext_info_len_; }

  [[nodiscard]] bool Parse(const std::string &ext_info);

  [[nodiscard]] bool UpdateInputShapeAndType(uint32_t input_index, const NotNull<AnfNodePtr> &anf_node);

  [[nodiscard]] bool UpdateOutputShapeAndType(uint32_t output_index, const NotNull<AnfNodePtr> &anf_node);

  [[nodiscard]] bool GetOutputShapeAndType(uint32_t output_index, NotNull<std::vector<int64_t> *> shape,
                                           NotNull<TypeId *> data_type);

  [[nodiscard]] bool UpdateEventId(const uint32_t event_id) const;
  [[nodiscard]] bool UpdateSessionInfoId(const uint64_t session_id) const;

 private:
  [[nodiscard]] bool ParseExtShapeType(const AicpuExtInfo &aicpu_ext_info) const;
  [[nodiscard]] bool ParseExtInputShape(AicpuExtInfo *aicpu_ext_info);
  [[nodiscard]] bool ParseExtOutputShape(AicpuExtInfo *aicpu_ext_info);
  [[nodiscard]] bool ParseExtSessionInfo(AicpuExtInfo *aicpu_ext_info);
  [[nodiscard]] bool ParseExtAsyncWait(AicpuExtInfo *aicpu_ext_info);

  [[nodiscard]] static bool UpdateShapeAndType(const std::vector<int64_t> &shape,
                                               NotNull<AicpuShapeAndType *> shape_and_type);

  static void GetShapeAndType(const NotNull<const AicpuShapeAndType *> &shape_and_type,
                              const NotNull<std::vector<int64_t> *> &shape, const NotNull<TypeId *> &data_type);

  bool GenerateKernelId() const;

  const std::string node_name_;
  const uint32_t input_num_;
  const uint32_t output_num_;
  ::ge::UnknowShapeOpType unknown_type_;
  size_t ext_info_len_;

  std::unique_ptr<uint8_t[]> ext_info_;
  std::vector<AicpuShapeAndType *> input_shape_and_type_;
  std::vector<AicpuShapeAndType *> output_shape_and_type_;
  AsyncWaitInfo *async_wait_ = nullptr;
  AicpuSessionInfo *session_info_ = nullptr;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AICPU_EXT_INFO_HANDLE_H_
