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

#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_ANFNODE_TENSOR_ADAPTER_H_
#define MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_ANFNODE_TENSOR_ADAPTER_H_
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include <vector>
#include "src/infer/tensor.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "include/api/status.h"

namespace mindspore {
namespace lite {
class TensorAdapter;
using TensorAdapterPtr = std::shared_ptr<TensorAdapter>;
class TensorAdapter {
 public:
  explicit TensorAdapter(std::string name) : name_(std::move(name)) {}
  virtual ~TensorAdapter() {
    if (own_data_) {
      free(data_);
    }
  }

  InferTensor *ToTensor();

  static TensorAdapterPtr Create(const ParameterPtr &param_node, Format format = DEFAULT_FORMAT);
  static TensorAdapterPtr Create(const ValueNodePtr &value_node, Format format = DEFAULT_FORMAT);
  static TensorAdapterPtr Create(const mindspore::abstract::AbstractTensorPtr &abstract,
                                 Format format = DEFAULT_FORMAT);
  static TensorAdapterPtr Create(const mindspore::abstract::AbstractBasePtr &abstract, Format format = DEFAULT_FORMAT);

  static std::vector<std::unique_ptr<InferTensor>> CreateTensorsFromAbstract(const AbstractBasePtr &abstract,
                                                                             Format format = Format::DEFAULT_FORMAT);
  static std::vector<InferTensor *> Convert2Tensor(const CNodePtr &cnode, Format format = DEFAULT_FORMAT);
  static InferTensor *Convert2Tensor(const ParameterPtr &param_node, Format format = DEFAULT_FORMAT);
  static InferTensor *Convert2Tensor(const ValueNodePtr &value_node, Format format = DEFAULT_FORMAT);
  static InferTensor *Convert2Tensor(const mindspore::abstract::AbstractTensorPtr &abstract,
                                     Format format = DEFAULT_FORMAT);
  static InferTensor *Convert2Tensor(const mindspore::abstract::AbstractBasePtr &abstract,
                                     Format format = DEFAULT_FORMAT);

  static StatusCode GetDTAndShapeFromAbTensor(const mindspore::abstract::AbstractTensorPtr &abstract, TypeId *data_type,
                                              ShapeVector *shape_vector);
  static StatusCode SetDTAndShapeFromAbTensor(const TypeId &data_type, const ShapeVector &shape,
                                              const mindspore::abstract::AbstractTensorPtr &abstract);
  static StatusCode SetDTAndShapeFromAbTensor(const TypeId &data_type, const std::vector<int> &shape,
                                              const mindspore::abstract::AbstractTensorPtr &abstract);

  static bool SetDTAndShapeFromAbTensorToLiteTensor(const AbstractBasePtr &abstract, InferTensor *tensor);
  static bool SetDTAndShapeFromLiteTensorToAbTensor(const InferTensor &tensor, const AbstractBasePtr &abstract);

 private:
  static StatusCode GetDTAndShapeFromParameter(const ParameterPtr &param_node, TypeId *data_type, ShapeVector *shape);

  static TensorAdapterPtr CreateFromTensorValueNode(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromInt32ImmValue(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromInt64ImmValue(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromBoolImmValue(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromNumberTypeValue(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromIntSequenceValue(const ValueNodePtr &value_node);

 public:
  Format format_{DEFAULT_FORMAT};
  TensorCompressionType compress_type_ = TensorCompressionType::kNoCompression;
  TypeId data_type_{kTypeUnknown};
  bool is_const_{false};
  ShapeVector shape_{};
  void *data_{nullptr};
  size_t data_len_{0};
  bool own_data_{true};
  std::string name_;
};
}  // namespace lite
}  // namespace mindspore

#endif
