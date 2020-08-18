/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
#define MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
#include <string>
#include <set>
#include <vector>
#include "src/ir/tensor.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"

#ifdef PRIMITIVE_WRITEABLE
#include "schema/inner/model_generated.h"
using OriginPrimitive = mindspore::schema::PrimitiveT;
#else
#include "schema/model_generated.h"
using OriginPrimitive = mindspore::schema::Primitive;
#endif

namespace mindspore {
namespace lite {
constexpr uint32_t kSingleNum = 1;
constexpr uint32_t kDoubleNum = 2;
constexpr uint32_t kMultiNum = 3;
constexpr uint32_t kDimension_4d = 4;

const std::set<int> kSupportDataType = {kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeFloat32};

// #if LITE_OPTIMIZE
class PrimitiveC {
 public:
  PrimitiveC() = default;

  explicit PrimitiveC(OriginPrimitive *primitive) : primitive(primitive) {}

  static PrimitiveC *CreatePrimitive(OriginPrimitive *primitive);

  virtual ~PrimitiveC() {}

  bool GetInferFlag() const;

  void SetInferFlag(bool flag);

  virtual int InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_);

  int Type() const;

 protected:
  OriginPrimitive *primitive;
  bool infer_flag_ = true;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
