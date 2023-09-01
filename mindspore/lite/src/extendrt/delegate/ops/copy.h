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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_OPS_COPY_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_OPS_COPY_H_
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "ops/base_operator.h"
#include "include/api/data_type.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCopy = "Copy";
constexpr auto kCopyFormat = "copy_format";

/// \brief Custom defined user-defined operator prototype.
class MIND_API Copy : public BaseOperator {
 public:
  enum CopyFormatType : int {
    NONE = 0,
    HOST_DEVICE,
    DEVICE_HOST,
  };

  MIND_API_BASE_MEMBER(Copy);
  /// \brief Constructor.
  Copy() : BaseOperator(kNameCopy) {}
  void set_copy_format(CopyFormatType format);
  int get_copy_format() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_OPS_COPY_H_
