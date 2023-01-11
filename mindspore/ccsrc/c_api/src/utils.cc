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

#include "c_api/src/utils.h"

void ConvertConstScalarInputToTensor(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (!input_node->isa<ValueNodeImpl>()) {
    return;
  }
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ScalarImpl>()) {
    return;
  }
  TensorPtr tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  if (tensor_ptr == nullptr) {
    MS_LOG(WARNING) << "Create tensor of" << input_node->DebugString() << "failed";
    return;
  }
  value_node->set_value(tensor_ptr);
  value_node->set_abstract(tensor_ptr->ToAbstract());
}

std::vector<TensorPtr> ConvertOutputToTensor(const mindspore::BaseRef &output) {
  std::vector<TensorPtr> ref_outputs{};
  if (mindspore::utils::isa<mindspore::VectorRef>(output)) {
    auto vec_ref = mindspore::utils::cast<mindspore::VectorRef>(output);
    for (const auto &item : vec_ref) {
      // for multiple outputs, ascend will return a VectorRef of VectorRef.
      const std::vector<TensorPtr> &item_out = ConvertOutputToTensor(item);
      (void)ref_outputs.insert(ref_outputs.end(), item_out.begin(), item_out.end());
    }
  } else if (mindspore::utils::isa<TensorPtr>(output)) {
    auto tensor = std::dynamic_pointer_cast<TensorImpl>(output.copy());
    tensor->data_sync();
    ref_outputs.push_back(tensor);
  } else if (mindspore::utils::isa<ScalarPtr>(output)) {
    auto value = mindspore::utils::cast<ScalarPtr>(output);
    auto tensor = ScalarToTensor(value->cast<ScalarPtr>());
    ref_outputs.push_back(tensor);
  } else {
    MS_LOG(ERROR) << "Convert output to tensor failed, unrecognized output type: " << output.ToString();
  }
  return ref_outputs;
}

AbstractBasePtr GetAbstract(const TypePtr &type_ptr, const int64_t shape[], size_t shape_size, bool is_param) {
  if (shape == nullptr) {
    if (shape_size == 0) {
      if (is_param) {
        ShapeVector shape_vec{1};
        return std::make_shared<AbstractTensorImpl>(type_ptr, shape_vec);
      }
      return std::make_shared<AbstractScalarImpl>(type_ptr);
    } else {
      MS_LOG(ERROR) << "Input Handle [shape_size] should >= 0.";
      return nullptr;
    }
  }
  if (shape[0] == 0 && shape_size == 1) {
    ShapeVector shape_vec;
    return std::make_shared<AbstractTensorImpl>(type_ptr, shape_vec);
  }
  ShapeVector shape_vec(shape, shape + shape_size);
  return std::make_shared<AbstractTensorImpl>(type_ptr, shape_vec);
}
