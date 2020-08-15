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
#include "ir/primitive.h"
#include "ir/value.h"

#ifdef PRIMITIVE_WRITEABLE
#include "schema/inner/model_generated.h"
#else
#include "schema/model_generated.h"
#endif
namespace mindspore {
constexpr uint32_t kSingleNum = 1;
constexpr uint32_t kDoubleNum = 2;
constexpr uint32_t kMultiNum = 3;
constexpr uint32_t kDimension_4d = 4;
enum NCHW_SHAPE { NCHW_N = 0, NCHW_C = 1, NCHW_H = 2, NCHW_W = 3 };
enum NHWC_SHAPE { NHWC_N = 0, NHWC_H = 1, NHWC_W = 2, NHWC_C = 3 };
enum HWCK_SHAPE { HWCK_H = 0, HWCK_W = 1, HWCK_C = 2, HWCK_K = 3 };
enum HWKC_SHAPE { HWKC_H = 0, HWKC_W = 1, HWKC_K = 2, HWKC_C = 3 };
enum KCHW_SHAPE { KCHW_K = 0, KCHW_C = 1, KCHW_H = 2, KCHW_W = 3 };
enum CKHW_SHAPE { CKHW_C = 0, CKHW_K = 1, CKHW_H = 2, CKHW_W = 3 };
enum CHWK_SHAPE { CHWK_C = 0, CHWK_H = 1, CHWK_W = 2, CHWK_K = 3 };
enum KHWC_SHAPE { KHWC_K = 0, KHWC_H = 1, KHWC_W = 2, KHWC_C = 3 };

const std::set<int> kSupportDataType = {kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeFloat32};

class PrimitiveC : public Primitive {
 public:
  explicit PrimitiveC(const std::string &name) : Primitive(name) {}

#ifdef PRIMITIVE_WRITEABLE
  explicit PrimitiveC(schema::PrimitiveT *primitive) : Primitive(""), primitive(primitive) {}
#else
  explicit PrimitiveC(schema::Primitive *primitive) : Primitive(""), primitive(primitive) {}
#endif
  static Primitive *CreatePrimitive(schema::Primitive *primitive);
  virtual ~PrimitiveC() {}
  const bool GetInferFlag() const { return this->infer_flag_; }
  void SetInferFlag(bool flag) { this->infer_flag_ = flag; }
  virtual int InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) = 0;

 protected:
#ifdef PRIMITIVE_WRITEABLE
  schema::PrimitiveT *primitive;
#else
  schema::Primitive *primitive;
#endif
  bool infer_flag_ = true;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
