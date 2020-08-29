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
constexpr int kAnfPopulaterOne = 1;
constexpr int kAnfPopulaterTwo = 2;
constexpr int kAnfPopulaterThree = 3;
class PrimitiveC : public mindspore::Primitive {
 public:
  // Argument primitive is deliverd into PrimitiveC and will be deleted in ~PrimitiveC().
  // Caller should not delete primitive.
  explicit PrimitiveC(schema::PrimitiveT *primitive) : Primitive(""), primitive_(primitive) {}

  explicit PrimitiveC(const Primitive &prim) : Primitive(prim) {}

  // Argument primitive is deliverd into PrimitiveC and will be deleted in ~PrimitiveC().
  // Caller should not delete primitive.
  explicit PrimitiveC(const std::string &name, schema::PrimitiveT *primitive)
      : Primitive(name), primitive_(primitive) {}

  PrimitiveC() : Primitive(""), primitive_(nullptr) {}

  MS_DECLARE_PARENT(PrimitiveC, Primitive);

  ~PrimitiveC() override { delete this->primitive_; }

  int Type() const;

  schema::PrimitiveT *GetPrimitiveT() const;

  void ClearPrimitiveT();

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

  static PrimitiveC *UnPackFromSchemaPrimitive(mindspore::schema::Primitive *primitive) {
    return UnPackFromSchemaPrimitiveT(primitive->UnPack());
  }

  static PrimitiveC *UnPackFromSchemaPrimitiveT(mindspore::schema::PrimitiveT *primitive);

  static std::shared_ptr<PrimitiveC> UnPackFromPrimitive(const Primitive &prim, const std::vector<AnfNodePtr> &inputs);

 protected:
  virtual int UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) { return RET_ERROR; }

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

  virtual ~PrimitiveC() { free(this->primitive_buf_); }

  static PrimitiveC *UnPackFromSchemaPrimitive(const schema::Primitive *primitive);

  bool GetInferFlag() const;

  void SetInferFlag(bool flag);

  virtual int InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs);

  int Type() const;

 protected:
  template <typename T, typename = std::enable_if<std::is_base_of<PrimitiveC, T>::value>>
  static PrimitiveC *NewPrimitiveC(const schema::Primitive *primitive) {
    auto primc = new T();
    if (primc == nullptr) {
      MS_LOG(ERROR) << "new PrimitiveC failed";
      return nullptr;
    }
    auto ret = primc->UnPackSchemaPrimitive(primitive);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "UnPackSchemaPrimitive failed";
      return nullptr;
    }
    return primc;
  }

  int UnPackSchemaPrimitive(const schema::Primitive *primitive) {
    flatbuffers::FlatBufferBuilder fbb(1024);
    if (UnPackToFlatBuilder(primitive, &fbb) != RET_OK) {
      MS_LOG(ERROR) << "UnPackToFlatBuilder failde";
      fbb.Clear();
      return RET_ERROR;
    }
    auto buf = fbb.GetBufferPointer();
    if (buf == nullptr) {
      MS_LOG(ERROR) << "GetBufferPointer return nullptr";
      fbb.Clear();
      return RET_ERROR;
    }
    primitive_buf_ = reinterpret_cast<char *>(malloc(fbb.GetSize()));
    if (primitive_buf_ == nullptr) {
      MS_LOG(ERROR) << "malloc primitive_buf_ failed";
      fbb.Clear();
      return RET_ERROR;
    }
    memcpy(primitive_buf_, buf, fbb.GetSize());
    this->primitive_ = flatbuffers::GetRoot<schema::Primitive>(primitive_buf_);
    fbb.Clear();
    return RET_OK;
  }

  virtual int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
    return RET_ERROR;
  }

 protected:
  const schema::Primitive *primitive_ = nullptr;
  char *primitive_buf_ = nullptr;
  bool infer_flag_ = true;
};
#endif
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
