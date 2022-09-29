/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_NNAPI_OP_NNAPI_OP_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_NNAPI_OP_NNAPI_OP_H_

#include <string>
#include <vector>
#include <utility>
#include "include/api/kernel.h"
#include "include/api/data_type.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/litert/delegate/nnapi/NeuralNetworksTypes.h"
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "schema/ops_generated.h"

namespace mindspore {
namespace lite {
struct MSTensorInfo {
  std::string name_;
  DataType type_;
  std::vector<int64_t> shape_;
  void *data_;
  size_t data_len_;
};
class NNAPIOp {
 public:
  explicit NNAPIOp(const std::string &name, const schema::Primitive *primitive,
                   std::vector<mindspore::MSTensor> in_tensors, std::vector<mindspore::MSTensor> out_tensors,
                   schema::QuantType quant_type)
      : op_name_(name),
        op_primitive_(primitive),
        in_tensors_(std::move(in_tensors)),
        out_tensors_(std::move(out_tensors)),
        quant_type_(quant_type) {
    if (primitive != nullptr) {
      this->type_ = primitive->value_type();
    }
  }

  virtual ~NNAPIOp() {
    for (auto tensor : op_attribute_tensors_) {
      if (tensor != nullptr) {
        delete tensor;
        tensor = nullptr;
      }
    }
  }

  virtual bool IsSupport() = 0;
  virtual int InitParams() = 0;
  virtual int ConvertInOutQuantSymmToASymm();
  virtual int AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) = 0;
  int InitNNAPIOpInOut(const std::vector<mindspore::MSTensor> &all_tensors);

  const std::vector<mindspore::MSTensor> &inputs() { return this->in_tensors_; }
  const std::vector<mindspore::MSTensor> &outputs() { return this->out_tensors_; }
  void set_inputs(const std::vector<mindspore::MSTensor> &inputs) { this->in_tensors_ = inputs; }
  void set_outputs(const std::vector<mindspore::MSTensor> &outputs) { this->out_tensors_ = outputs; }

  const std::vector<NNAPIOp *> &in_ops() { return this->in_ops_; }
  const std::vector<NNAPIOp *> &out_ops() { return this->out_ops_; }
  void set_in_ops(const std::vector<NNAPIOp *> &in_ops) { this->in_ops_ = in_ops; }
  void set_out_ops(const std::vector<NNAPIOp *> &out_ops) { this->out_ops_ = out_ops; }

  const std::string name() { return op_name_; }
  schema::QuantType get_quant_type() { return quant_type_; }

 protected:
  template <typename T>
  int AddScalarToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors,
                            std::string name, DataType type, T value) {
    auto tensor = MSTensor::CreateTensor(name, type, {}, &value, sizeof(T));
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return RET_ERROR;
    }
    if (AddNNAPIOperand(nnapi_model, *tensor, static_cast<int>(all_tensors->size()), 0, true) != RET_OK) {
      MS_LOG(ERROR) << "Add NNAPI operand failed.";
      delete tensor;
      return RET_ERROR;
    }
    input_indices_.push_back(all_tensors->size());
    all_tensors->push_back(*tensor);
    op_attribute_tensors_.push_back(tensor);
    return RET_OK;
  }
  int AddTensorToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors,
                            MSTensorInfo data_info);

  std::string op_name_;
  const schema::Primitive *op_primitive_ = nullptr;
  std::vector<mindspore::MSTensor> in_tensors_;
  std::vector<mindspore::MSTensor> out_tensors_;
  schema::PrimitiveType type_ = schema::PrimitiveType_NONE;
  schema::QuantType quant_type_ = schema::QuantType_QUANT_NONE;

  std::vector<NNAPIOp *> in_ops_;
  std::vector<NNAPIOp *> out_ops_;

  std::vector<uint32_t> input_indices_;
  std::vector<uint32_t> output_indices_;
  std::vector<MSTensor *> op_attribute_tensors_;
};

typedef NNAPIOp *(*NNAPIGetOp)(const std::string &name, const schema::Primitive *primitive,
                               const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors, schema::QuantType quant_type);

template <class T>
NNAPIOp *GetNNAPIOp(const std::string &name, const schema::Primitive *primitive,
                    const std::vector<mindspore::MSTensor> &in_tensors,
                    const std::vector<mindspore::MSTensor> &out_tensors, schema::QuantType quant_type) {
  MS_ASSERT(primitive != nullptr);
  auto *op = new (std::nothrow) T(name, primitive, in_tensors, out_tensors, quant_type);
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is nullptr.";
    return nullptr;
  }
  auto ret = op->InitParams();
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "NPU op init failed.";
    delete op;
    return nullptr;
  }
  if (!op->IsSupport()) {
    MS_LOG(WARNING) << "NNAPI op is not supported.";
    delete op;
    return nullptr;
  }
  return op;
}

class NNAPICommon : public NNAPIOp {
 public:
  NNAPICommon(const std::string &name, const schema::Primitive *primitive,
              const std::vector<mindspore::MSTensor> &in_tensors, const std::vector<mindspore::MSTensor> &out_tensors,
              schema::QuantType quant_type)
      : NNAPIOp(name, primitive, in_tensors, out_tensors, quant_type) {}

  ~NNAPICommon() override {}

  bool IsSupport() override { return true; };
  int InitParams() override { return RET_OK; };
  int AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_NNAPI_OP_NNAPI_OP_H_
