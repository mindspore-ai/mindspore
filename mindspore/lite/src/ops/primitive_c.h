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

#ifndef MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
#define MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
#include <string>
#include <set>
#include <vector>
#include <memory>
#ifdef PRIMITIVE_WRITEABLE
#include "ir/primitive.h"
#include "schema/inner/model_generated.h"
#else
#include "schema/model_generated.h"
#endif

#include "src/ir/tensor.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace lite {
constexpr uint32_t kSingleNum = 1;
constexpr uint32_t kDoubleNum = 2;
constexpr uint32_t kMultiNum = 3;
constexpr uint32_t kDimension_4d = 4;

const std::set<int> kSupportDataType = {kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat16};

#ifdef PRIMITIVE_WRITEABLE
class PrimitiveC : public mindspore::Primitive {
 public:
  explicit PrimitiveC(schema::Primitive *primitive) : Primitive("") { this->primitive_ = primitive->UnPack(); }

  explicit PrimitiveC(schema::PrimitiveT *primitive) : Primitive(""), primitive_(primitive) {}

  explicit PrimitiveC(const Primitive &prim) : Primitive(prim) {}

  explicit PrimitiveC(const std::string &name, schema::PrimitiveT *primitive)
      : Primitive(name), primitive_(primitive) {}

  MS_DECLARE_PARENT(PrimitiveC, Primitive);

  ~PrimitiveC() override = default;

  int Type() const;

  //  static PrimitiveC *UnPackFromPrimitive(const Primitive &prim);

  schema::PrimitiveT *GetPrimitiveT() const;

  void SetPrimitiveT(schema::PrimitiveT *prim);

  bool operator==(const Value &rhs) const {
    if (rhs.isa<PrimitiveC>()) {
      auto other_prim = static_cast<const PrimitiveC &>(rhs);
      auto a = this->primitive_->value.type;
      auto b = other_prim.primitive_->value.type;
      return a == b;
    } else {
      return false;
    }
  }

  void SetInputQuantParam(const std::vector<std::vector<schema::QuantParamT>> &input_quant_param);

  void SetOutputQuantParam(const std::vector<std::vector<schema::QuantParamT>> &output_quant_param);

  void ClearInputOutputQuantParam();

  void AddInputQuantParam(std::vector<schema::QuantParamT> quant_param);

  std::vector<std::vector<schema::QuantParamT>> GetInputQuantParams() const;

  void AddOutputQuantParam(std::vector<schema::QuantParamT> quant_param);

  std::vector<std::vector<schema::QuantParamT>> GetOutputQuantParams() const;

  void SetQuantType(schema::QuantType quant_type);

  schema::QuantType GetQuantType() const;

  virtual int InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_);

  bool GetInferFlag() const;

  void SetInferFlag(bool flag);

  static PrimitiveC *CreatePrimitive(mindspore::schema::Primitive *primitive);

 protected:
  //  virutal PrimitiveC *UnPackAttr(const Primitive &prim) = 0;

 protected:
  schema::PrimitiveT *primitive_ = nullptr;
  std::vector<std::vector<schema::QuantParamT>> input_quant_param_;
  std::vector<std::vector<schema::QuantParamT>> output_quant_param_;
  schema::QuantType quant_type_{schema::QuantType_QUANT_NONE};
  bool infer_flag_ = true;
};
std::shared_ptr<PrimitiveC> GetReturnPrim();

std::shared_ptr<PrimitiveC> GetMakeTuplePrim();

std::shared_ptr<PrimitiveC> GetTupleGetItemPrim();



#else
class PrimitiveC {
 public:
  PrimitiveC() = default;

  explicit PrimitiveC(schema::Primitive *primitive) : primitive_(primitive) {}

  virtual ~PrimitiveC() = default;

  bool GetInferFlag() const;

  void SetInferFlag(bool flag);

  virtual int InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs);

  int Type() const;

 protected:
  schema::Primitive *primitive_ = nullptr;
  bool infer_flag_ = true;
};
#endif
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
