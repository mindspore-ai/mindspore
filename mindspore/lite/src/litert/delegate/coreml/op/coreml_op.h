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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_COREML_OP_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_COREML_OP_
#include <utility>
#include <vector>
#include <string>
#include <set>
#include <memory>
#include <unordered_map>
#include "proto/Model.pb.h"
#include "proto/NeuralNetwork.pb.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;
namespace mindspore::lite {
inline const std::vector<int> NHWC2NCHW_PERM = {0, 3, 1, 2};
inline const std::vector<int> NCHW2NHWC_PERM = {0, 2, 3, 1};
enum COREML_WEIGHT_SHAPE { ML_WT_COUT = 0, ML_WT_CIN = 1, ML_WT_H = 2, ML_WT_W = 3 };
enum MSLITE_WEIGHT_SHAPE { MS_WT_COUT = 0, MS_WT_H = 1, MS_WT_W = 2, MS_WT_CIN = 3 };
enum PAD { PAD_UP = 0, PAD_DOWN = 1, PAD_LEFT = 2, PAD_RIGHT = 3 };
constexpr int REPEAT_TIMES2 = 2;
class CoreMLOp {
 public:
  CoreMLOp(const schema::Primitive *primitive, std::vector<mindspore::MSTensor> in_tensors,
           std::vector<mindspore::MSTensor> out_tensors, std::string name)
      : op_primitive_(primitive),
        in_tensors_(std::move(in_tensors)),
        out_tensors_(std::move(out_tensors)),
        name_(std::move(name)) {
    if (primitive != nullptr) {
      type_ = primitive->value_type();
    }
  }

  // the op will be managed by coreml model, no need to manually deconstruct
  virtual ~CoreMLOp() = default;

  virtual int IsSupport() { return RET_OK; }

  virtual int Init();

  virtual int InitParams() { return RET_OK; }

  virtual int HandleAxis() { return RET_OK; }

  virtual int BuildLayer() { return RET_OK; }

  // override this method if the op has tensor which does not need to add to graph，e.g.，const tensor.
  virtual void SetMLOpInOut();

  // Transfer the ownership of op to coreml model; Multiple layers are possible to be build for one op, thus using
  // vector as return.
  virtual std::vector<CoreML::Specification::NeuralNetworkLayer *> GetLayers();

  virtual int SetActivation(schema::ActivationType act_type);

  virtual int SetPadding(std::vector<int> pad_list);

  virtual int SetConstInput(const mindspore::MSTensor &in_tensor);

  void set_inputs(const std::vector<mindspore::MSTensor> &in_tensors) { this->in_tensors_ = in_tensors; }

  void set_input(const mindspore::MSTensor &in_tensor, int index) {
    MS_ASSERT(static_cast<size_t>(index) < in_tensors_.size());
    this->in_tensors_[index] = in_tensor;
  }

  void set_outputs(const std::vector<mindspore::MSTensor> &out_tensors) { this->out_tensors_ = out_tensors; }

  const std::vector<mindspore::MSTensor> &inputs() { return this->in_tensors_; }

  const std::vector<mindspore::MSTensor> &outputs() { return this->out_tensors_; }

  void set_in_ops(const std::vector<CoreMLOp *> &in_ops) { this->in_ops_ = in_ops; }

  void set_out_ops(const std::vector<CoreMLOp *> &out_ops) { this->out_ops_ = out_ops; }

  const std::vector<CoreMLOp *> &in_ops() const { return this->in_ops_; }

  const std::vector<CoreMLOp *> &out_ops() const { return this->out_ops_; }

  schema::PrimitiveType type() const { return type_; }

  std::string name() const { return this->name_; }

  void set_name(const std::string &name) { this->name_ = name; }

 protected:
  const schema::Primitive *op_primitive_ = nullptr;
  std::vector<mindspore::MSTensor> in_tensors_;
  std::vector<mindspore::MSTensor> out_tensors_;
  std::vector<CoreMLOp *> in_ops_;
  std::vector<CoreMLOp *> out_ops_;
  schema::PrimitiveType type_ = schema::PrimitiveType_NONE;
  std::string name_;
  std::unique_ptr<CoreML::Specification::NeuralNetworkLayer> op_ = nullptr;
  std::unique_ptr<CoreML::Specification::NeuralNetworkLayer> pad_op_ = nullptr;
  std::unique_ptr<CoreML::Specification::NeuralNetworkLayer> act_op_ = nullptr;
  std::unordered_map<std::string, std::unique_ptr<CoreML::Specification::NeuralNetworkLayer>> const_ops_ = {};
};

typedef CoreMLOp *(*CoreMLGetOp)(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                                 const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name);

template <class T>
CoreMLOp *GetCoreMLOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name) {
  auto shape = out_tensors.front().Shape();
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    MS_LOG(ERROR) << "CoreML does not support runtime inference shape.";
    return nullptr;
  }
  auto *op = new (std::nothrow) T(primitive, in_tensors, out_tensors, name);
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is nullptr.";
    return nullptr;
  }
  auto ret = op->IsSupport();
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "CoreML op is not supported.";
    delete op;
    return nullptr;
  }
  ret = op->Init();
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "CoreML op init failed.";
    delete op;
    return nullptr;
  }
  return op;
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_COREML_OP_
