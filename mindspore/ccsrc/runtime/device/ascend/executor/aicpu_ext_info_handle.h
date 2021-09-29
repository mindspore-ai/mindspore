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
#include "backend/kernel_compiler/aicpu/aicpu_util.h"
#include "utils/contract.h"

namespace mindspore {
namespace device {
namespace ascend {
// for unknown shape op type
enum UnknowShapeOpType {
  DEPEND_IN_SHAPE = 1,     // op out shape get by input shape
  DEPEND_CONST_VALUE = 2,  // op out shape get by const op value
  DEPEND_SHAPE_RANGE = 3,  // op out shape get by range
  DEPEND_COMPUTE = 4       // op out shape get by totally computing
};

using AicpuShapeAndType = kernel::ShapeAndType;
using AicpuExtInfo = kernel::ExtInfo;

class AicpuExtInfoHandler {
 public:
  AicpuExtInfoHandler(std::string node_name, uint32_t input_num, uint32_t output_num, UnknowShapeOpType unknown_type)
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

 private:
  [[nodiscard]] bool ParseExtShapeType(AicpuExtInfo *aicpu_ext_info);
  [[nodiscard]] bool ParseExtInputShape(AicpuExtInfo *aicpu_ext_info);
  [[nodiscard]] bool ParseExtOutputShape(AicpuExtInfo *aicpu_ext_info);

  [[nodiscard]] static bool UpdateShapeAndType(const std::vector<int64_t> &shape,
                                               NotNull<AicpuShapeAndType *> shape_and_type);

  static void GetShapeAndType(NotNull<const AicpuShapeAndType *> shape_and_type, NotNull<std::vector<int64_t> *> shape,
                              NotNull<TypeId *> data_type);

 private:
  const std::string node_name_;
  const uint32_t input_num_;
  const uint32_t output_num_;
  UnknowShapeOpType unknown_type_;
  size_t ext_info_len_;

  std::unique_ptr<uint8_t[]> ext_info_;
  std::vector<AicpuShapeAndType *> input_shape_and_type_;
  std::vector<AicpuShapeAndType *> output_shape_and_type_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AICPU_EXT_INFO_HANDLE_H_
