/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/runtime/meta_tensor_utils.h"

#include <memory>

#include "device_address/device_address.h"
#include "ir/dtype/type_id.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore::runtime {
namespace {
void AppendMetaOutputTensors(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(outputs);
  if (abstract->isa<abstract::AbstractSequence>()) {
    const auto &seq = abstract->cast<abstract::AbstractSequencePtr>();
    for (const auto &element : seq->elements()) {
      AppendMetaOutputTensors(element, outputs);
    }
    return;
  }

  TypeId type_id;
  ShapeVector shape_vector;
  if (abstract->isa<abstract::AbstractTensor>()) {
    const auto &abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
    const auto &shape = abstract_tensor->GetShapeTrack();
    const auto &type = abstract_tensor->element()->GetTypeTrack();
    if (!shape->isa<abstract::Shape>()) {
      MS_LOG(EXCEPTION) << "AbstractTensor shape is not static: " << shape->ToString();
    }
    const auto &shape_ptr = shape->cast<abstract::ShapePtr>();
    shape_vector = shape_ptr->shape();
    if (IsDynamic(shape_vector)) {
      MS_LOG(EXCEPTION) << "The operator does not support meta device because the output shape is dynamic: "
                        << shape->ToString();
    }
    type_id = type->type_id();
  } else if (abstract->isa<abstract::AbstractScalar>()) {
    const auto &scalar = abstract->cast<abstract::AbstractScalarPtr>();
    type_id = scalar->GetTypeTrack()->type_id();
  } else {
    MS_LOG(EXCEPTION) << "Not support abstract " << abstract->ToString();
  }

  outputs->emplace_back(CreateMetaOutputTensor(type_id, shape_vector));
}
}  // namespace

tensor::TensorPtr CreateMetaOutputTensor(const TypeId &type_id, const ShapeVector &shape_vector) {
  auto data_size = SizeOf(shape_vector) * abstract::TypeIdSize(type_id);
  auto device_address = std::make_shared<device::DeviceAddress>(nullptr, data_size, device::DeviceType::kMeta, 0);
  auto output_tensor = std::make_shared<tensor::Tensor>(type_id, shape_vector, device_address);
  output_tensor->set_need_pipeline_sync(true);
  MS_LOG(DEBUG) << "Create meta output tensor " << output_tensor->ToString();
  return output_tensor;
}

void CreateMetaOutputTensors(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs) {
  AppendMetaOutputTensors(abstract, outputs);
}
}  // namespace mindspore::runtime
