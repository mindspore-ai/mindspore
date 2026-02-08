/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "include/backend/common/pass_manager/dynamic_shape_helper.h"

#include <memory>
#include <algorithm>
#include <stack>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "primitive/framework_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "utils/anf_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/infershape_functor.h"
#include "ops/op_def.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"
#include "tools/profiler/profiler.h"
#include "ir/anf.h"
#include "ir/functor.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace opt::dynamic_shape {
BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &op_name = primitive->name();
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelInferInner,
                                     op_name, true);
  if (primitive->HasAttr(kAttrInferShapeFunctor)) {
    auto functor = primitive->GetAttr(kAttrInferShapeFunctor)->cast<InferShapeFunctorPtr>();
    MS_EXCEPTION_IF_NULL(functor);
    return functor->InferShape(input_args);
  }
  auto shape_optional = abstract::InferShapeByFuncImpl(primitive, input_args, false);
  if (shape_optional.has_value()) {
    return shape_optional.value();
  }

  // The old register map for InferShape will be deleted in the future.
  auto found = abstract::GetPrimitiveInferImpl(primitive);
  if (found.has_value()) {
    auto infer = found.value();
    if (infer.IsImplInferShapeAndType()) {
      return infer.InferShape(primitive, input_args);
    }
  }
  MS_LOG(EXCEPTION) << "The InferShape function of [" << op_name << "] is not defined.";
}

void UpdateKernelTensorShape(const BaseShapePtr &base_shape,
                             const std::vector<kernel::KernelTensor *> &output_kernel_tensors) {
  MS_EXCEPTION_IF_NULL(base_shape);
  size_t output_num = output_kernel_tensors.size();
  if (output_num > 1) {
    auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape);
    const auto &shapes = sequence_shape->shape();
    if (shapes.size() != output_num) {
      MS_LOG(EXCEPTION) << "Invalid SequenceShape, expected elements number: " << output_num
                        << ", but got: " << shapes.size();
    }
    for (size_t i = 0; i < output_num; i++) {
      const auto &kernel_tensor = output_kernel_tensors[i];
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      kernel_tensor->SetShape(shapes[i]);
    }
  } else if (output_num == 1) {
    const auto &kernel_tensor = output_kernel_tensors[0];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    if ((kernel_tensor->type_id() != kObjectTypeTuple && kernel_tensor->type_id() != kObjectTypeList) &&
        sequence_shape != nullptr) {
      // For the operator prototype whose output is of type Tuple, the back-end operator is expanded as Tensors, and for
      // single-output scenarios, the InferShape result is TupleShape, and the back-end needs to expand it to
      // TensorShape. For example, the output of the split operator is only a Tensor scene.
      const auto &shapes = sequence_shape->shape();
      if (shapes.size() != 1) {
        MS_LOG(EXCEPTION) << "Invalid SequenceShape, expected elements number: " << 1 << ", but got: " << shapes.size();
      }

      kernel_tensor->SetShape(shapes[0]);
    } else {
      kernel_tensor->SetShape(base_shape);
    }
  }
}

abstract::AbstractBasePtr InferShapeAndType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &op_name = primitive->name();
  auto found = abstract::GetPrimitiveInferImpl(primitive);
  if (found.has_value()) {
    auto infer = found.value();
    if (infer.IsImplInferShapeAndType()) {
      return infer.InferShapeAndType(nullptr, primitive, input_args);
    }
  }
  MS_LOG(EXCEPTION) << "The InferShape function of [" << op_name << "] is not defined.";
}

void UpdateKernelTensorType(const TypePtr &type, const std::vector<kernel::KernelTensor *> &output_kernel_tensors) {
  MS_EXCEPTION_IF_NULL(type);
  if (output_kernel_tensors.size() != 1) {
    MS_LOG(EXCEPTION) << "Invalid output size:" << output_kernel_tensors.size();
  }

  const auto &kernel_tensor = output_kernel_tensors[0];
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  kernel_tensor->SetType(type);
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
