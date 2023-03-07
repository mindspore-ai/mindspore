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
#include <unordered_map>
#include <vector>
#include "src/extendrt/tensor.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "include/api/status.h"

namespace mindspore {
namespace infer {
class TensorAdapter;
using TensorAdapterPtr = std::shared_ptr<TensorAdapter>;
class TensorAdapter {
 public:
  TensorAdapter() = default;
  virtual ~TensorAdapter() {
    if (own_data_) {
      free(data_);
    }
  }

  Tensor *ToTensor(const std::string &tensor_name = "");

  static TensorAdapterPtr Create(const ParameterPtr &param_node);
  static TensorAdapterPtr Create(const ValueNodePtr &value_node);
  static TensorAdapterPtr Create(const mindspore::abstract::AbstractTensorPtr &abstract);
  static TensorAdapterPtr Create(const mindspore::abstract::AbstractBasePtr &abstract);

  static Tensor *Convert2Tensor(const ParameterPtr &param_node, const std::string &tensor_name = "");
  static Tensor *Convert2Tensor(const ValueNodePtr &value_node, const std::string &tensor_name = "");
  static Tensor *Convert2Tensor(const mindspore::abstract::AbstractTensorPtr &abstract,
                                const std::string &tensor_name = "");
  static Tensor *Convert2Tensor(const mindspore::abstract::AbstractBasePtr &abstract,
                                const std::string &tensor_name = "");

 private:
  static StatusCode GetDTAndShapeFromAbTensor(const mindspore::abstract::AbstractTensorPtr &abstract, TypeId *data_type,
                                              ShapeVector *shape_vector);

  static StatusCode GetDTAndShapeFromParameter(const ParameterPtr &param_node, TypeId *data_type,
                                               ShapeVector *shape_vector);

  static TensorAdapterPtr CreateFromTensorValueNode(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromInt32ImmValue(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromInt64ImmValue(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromBoolImmValue(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromNumberTypeValue(const ValueNodePtr &value_node);

  static TensorAdapterPtr CreateFromIntSequenceValue(const ValueNodePtr &value_node);

 public:
  Format format_{DEFAULT_FORMAT};
  TensorCompressionType compress_type_{kNoCompression};
  TypeId data_type_{kTypeUnknown};
  bool is_const_{false};
  std::vector<int64_t> shape_{};
  void *data_{nullptr};
  size_t data_len_{0};
  bool own_data_{true};
};
}  // namespace infer
}  // namespace mindspore

#endif
