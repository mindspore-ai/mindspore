/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_NPU_OP_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_NPU_OP_H_
#include <utility>
#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include "include/graph/graph.h"
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
namespace mindspore {
constexpr int NPU_SHAPE_SIZE = 4;
constexpr int REPEAT_TIMES2 = 2;

class NPUOp {
 public:
  NPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
        const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : inputs_(std::move(in_tensors)), outputs_(std::move(out_tensors)), name_(name) {
    if (primitive != nullptr) {
      type_ = primitive->value_type();
    }
  }

  virtual ~NPUOp() = default;

  virtual int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                        const std::vector<mindspore::MSTensor> &out_tensors) {
    return RET_ERROR;
  }

  virtual int Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensors) {
    return RET_ERROR;
  }

  virtual int SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors,
                           const std::vector<ge::Operator *> &npu_inputs) {
    return RET_ERROR;
  }

  virtual int SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors,
                           const std::vector<ge::Operator *> &npu_inputs,
                           const std::unordered_map<int, std::pair<ge::Operator *, int>> &index2_multi_out_index) {
    if (index2_multi_out_index.empty()) {
      return SetNPUInputs(in_tensors, out_tensors, npu_inputs);
    }
    MS_LOG(ERROR) << "The input operator of npu op: " << this->name() << " has multiple outputs. Override this method.";
    return RET_ERROR;
  }

  virtual ge::Operator *GetNPUOp() { return nullptr; }

  virtual int HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) { return RET_OK; }

  void set_inputs(const std::vector<mindspore::MSTensor> &in_tensors) { this->inputs_ = in_tensors; }

  void set_input(mindspore::MSTensor in_tensor, int index) {
    MS_ASSERT(index < inputs_.size());
    this->inputs_[index] = in_tensor;
  }

  void set_outputs(const std::vector<mindspore::MSTensor> &out_tensors) { this->outputs_ = out_tensors; }

  const std::vector<mindspore::MSTensor> &inputs() { return this->inputs_; }

  const std::vector<mindspore::MSTensor> &outputs() { return this->outputs_; }

  void set_in_ops(const std::vector<NPUOp *> &in_ops) { this->in_ops_ = in_ops; }

  void set_out_ops(const std::vector<NPUOp *> &out_ops) { this->out_ops_ = out_ops; }

  const std::vector<NPUOp *> &in_ops() const { return this->in_ops_; }

  const std::vector<NPUOp *> &out_ops() const { return this->out_ops_; }

  schema::PrimitiveType type() const { return type_; }

  std::string name() const { return this->name_; }

  void set_name(const std::string &name) { this->name_ = name; }

 protected:
  std::vector<mindspore::MSTensor> inputs_;
  std::vector<mindspore::MSTensor> outputs_;
  std::vector<NPUOp *> in_ops_;
  std::vector<NPUOp *> out_ops_;
  schema::PrimitiveType type_ = schema::PrimitiveType_NONE;
  std::string name_;
};

typedef NPUOp *(*NPUGetOp)(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors, std::string name);

template <class T>
NPUOp *GetNPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors, std::string name) {
  MS_ASSERT(primitive != nullptr);
  auto shape = out_tensors.front().Shape();
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    MS_LOG(ERROR) << "NPU does not support runtime inference shape.";
    return nullptr;
  }

  if (in_tensors[0].Shape().size() > NPU_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Npu does not support input tensor dims greater than 4";
    return nullptr;
  }

  std::set<schema::PrimitiveType> int32_lists = {schema::PrimitiveType_Cast, schema::PrimitiveType_StridedSlice,
                                                 schema::PrimitiveType_Reshape, schema::PrimitiveType_ReduceFusion};
  std::set<schema::PrimitiveType> bool_lists = {schema::PrimitiveType_Cast,       schema::PrimitiveType_LogicalAnd,
                                                schema::PrimitiveType_LogicalOr,  schema::PrimitiveType_LogicalNot,
                                                schema::PrimitiveType_TileFusion, schema::PrimitiveType_BroadcastTo};
  auto support_int32 = in_tensors[0].DataType() == DataType::kNumberTypeInt32 &&
                       find(int32_lists.begin(), int32_lists.end(), primitive->value_type()) != int32_lists.end();
  auto support_bool = in_tensors[0].DataType() == DataType::kNumberTypeBool &&
                      find(bool_lists.begin(), bool_lists.end(), primitive->value_type()) != bool_lists.end();
  if (in_tensors[0].DataType() != DataType::kNumberTypeFloat32 &&
      in_tensors[0].DataType() != DataType::kNumberTypeFloat16 && !support_int32 && !support_bool) {
    MS_LOG(ERROR) << "Npu does not support datatype " << static_cast<int>(in_tensors[0].DataType()) << " for op type "
                  << primitive->value_type();
    return nullptr;
  }

  auto *op = new (std::nothrow) T(primitive, in_tensors, out_tensors, name);
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is nullptr.";
    return nullptr;
  }
  auto ret = op->IsSupport(primitive, in_tensors, out_tensors);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "NPU op is not supported.";
    delete op;
    return nullptr;
  }
  ret = op->Init(primitive, in_tensors, out_tensors);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "NPU op init failed.";
    delete op;
    return nullptr;
  }
  return op;
}
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_NPU_OP_H_
