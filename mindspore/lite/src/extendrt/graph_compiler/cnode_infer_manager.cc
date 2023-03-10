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
#include "src/extendrt/graph_compiler/cnode_infer_manager.h"
#include <algorithm>
#include "abstract/abstract_value.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "src/extendrt/graph_compiler/anfnode_tensor_adapter.h"

namespace mindspore {
namespace infer {
bool SetDTAndShapeFromAbTensorToLiteTensor(const AbstractBasePtr &abstract, lite::Tensor *tensor) {
  if (!utils::isa<mindspore::abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(ERROR) << "The abstract should be tensor, but got abstract : " << abstract;
    return false;
  }
  ShapeVector shape_vector;
  TypeId data_type = kTypeUnknown;
  auto ret = infer::TensorAdapter::GetDTAndShapeFromAbTensor(
    utils::cast<mindspore::abstract::AbstractTensorPtr>(abstract), &data_type, &shape_vector);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get dtype and shape from abstract failed, abstract : " << abstract;
    return false;
  }
  std::vector<int32_t> int32_shape;
  std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(int32_shape),
                 [](const auto &shape) { return static_cast<int32_t>(shape); });
  tensor->set_data_type(data_type);
  tensor->set_shape(int32_shape);
  tensor->set_format(NHWC);
  return true;
}

int CNodeInferShape(const CNodePtr &cnode, const std::vector<lite::Tensor *> &outputs) {
  session::AnfRuntimeAlgorithm::InferShape(cnode);
  // sync cnode abstract info to Lite Tensor
  auto abstract = cnode->abstract();
  if (utils::isa<mindspore::abstract::AbstractSequencePtr>(abstract)) {
    auto elements = utils::cast<mindspore::abstract::AbstractSequencePtr>(abstract)->elements();
    if (elements.size() != outputs.size()) {
      MS_LOG(ERROR) << "The cnode output size: " << elements.size()
                    << " is not equal to lite tensors size: " << outputs.size();
      return lite::RET_ERROR;
    }
    for (size_t i = 0; i < elements.size(); i++) {
      if (!SetDTAndShapeFromAbTensorToLiteTensor(elements[i], outputs[i])) {
        MS_LOG(ERROR) << "Set tensor info from abstract failed, abstract : " << elements[i];
        return lite::RET_ERROR;
      }
    }
    return lite::RET_OK;
  }
  if (utils::isa<mindspore::abstract::AbstractTensorPtr>(abstract)) {
    if (!SetDTAndShapeFromAbTensorToLiteTensor(abstract, outputs[0])) {
      MS_LOG(ERROR) << "Set tensor info from abstract failed, abstract : " << abstract;
      return lite::RET_ERROR;
    }
    return lite::RET_OK;
  }
  MS_LOG(ERROR) << "Unsupported abstract: " << abstract;
  return lite::RET_ERROR;
}
}  // namespace infer
}  // namespace mindspore
