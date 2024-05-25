/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ops/batch_isend_irecv.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/other_ops.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(BatchISendIRecv, BaseOperator);
class BatchISendIRecvInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
    size_t isend_tensor_count = 0;
    MS_EXCEPTION_IF_NULL(input_args[0]);
    if (input_args[0]->isa<abstract::AbstractSequence>()) {
      isend_tensor_count = input_args[0]->cast<abstract::AbstractSequencePtr>()->size();
    } else {
      MS_LOG(EXCEPTION) << "BatchISendIRecv input must be tuple or list of tensors.";
    }

    auto recv_shape = GetValue<std::vector<std::vector<int64_t>>>(primitive->GetAttr("receive_shapes"));
    auto recv_shape_size = recv_shape.size();
    auto op_types_ = GetValue<std::vector<std::string>>(primitive->GetAttr("op_types"));
    size_t recv_shape_index = 0;
    size_t send_count = 0;
    std::vector<BaseShapePtr> out_shape;
    for (size_t i = 0; i < op_types_.size(); i++) {
      if (op_types_[i] == "irecv") {
        if (recv_shape_index > recv_shape_size - 1) {
          MS_LOG(EXCEPTION) << "BatchISendIRecv: the count of 'irecv' in 'op_types' not match 'receive_shapes'.";
        }
        auto ptr_ = std::make_shared<abstract::TensorShape>(ShapeVector(recv_shape[recv_shape_index]));
        out_shape.push_back(ptr_);
        recv_shape_index++;
      } else if (op_types_[i] == "isend") {
        auto ptr_ = std::make_shared<abstract::TensorShape>(ShapeVector({}));
        out_shape.push_back(ptr_);
        send_count++;
      } else {
        MS_LOG(EXCEPTION) << "BatchISendIRecv only support 'isend' or 'irecv', but got "
                          << "'" << op_types_[i] << "'.";
      }
    }
    if (isend_tensor_count != send_count) {
      MS_LOG(EXCEPTION) << "BatchISendIRecv input tensors count not match 'isend' in 'op_types'.";
    }
    return std::make_shared<abstract::TupleShape>(std::move(out_shape));
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();

    const std::set<TypePtr> default_target_dtypes = common_valid_types;
    const std::set<TypePtr> target_dtypes = common_valid_types_with_bool;
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);

    // flag to check different valid types on ascend
    auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);

    std::vector<TypePtr> out_types;
    auto recv_dtype_attr = prim->GetAttr("receive_dtypes");
    MS_EXCEPTION_IF_NULL(recv_dtype_attr);
    const std::vector<ValuePtr> &recv_dtypes = recv_dtype_attr->cast<ValueSequencePtr>()->value();
    auto recv_type_size = recv_dtypes.size();

    auto op_types_ = GetValue<std::vector<std::string>>(prim->GetAttr("op_types"));

    size_t isend_tensor_count = 0;
    MS_EXCEPTION_IF_NULL(input_args[0]);
    if (input_args[0]->isa<abstract::AbstractSequence>()) {
      isend_tensor_count = input_args[0]->cast<abstract::AbstractSequencePtr>()->size();
    } else {
      MS_LOG(EXCEPTION) << "BatchISendIRecv input must be tuple or list of tensors.";
    }
    auto elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();

    size_t recv_type_index = 0;
    size_t send_type_index = 0;

    for (size_t i = 0; i < op_types_.size(); i++) {
      TypePtr type_ptr = nullptr;
      if (op_types_[i] == "irecv") {
        if (recv_type_index > recv_type_size - 1) {
          MS_LOG(EXCEPTION) << "BatchISendIRecv: the count of 'irecv' in 'op_types' not match 'receive_types'.";
        }
        type_ptr = std::make_shared<TensorType>(recv_dtypes[recv_type_index]->cast<TypePtr>());
        recv_type_index++;
      } else if (op_types_[i] == "isend") {
        auto elemi_ptr = elements[send_type_index];
        MS_EXCEPTION_IF_NULL(elemi_ptr);
        auto elemi = elemi_ptr->cast<abstract::AbstractTensorPtr>();
        MS_EXCEPTION_IF_NULL(elemi);
        type_ptr = elemi->GetType();
        send_type_index++;
      } else {
        MS_LOG(EXCEPTION) << "HcclBatchISendIRecv only support 'isend' or 'irecv', but got "
                          << "'" << op_types_[i] << "'.";
      }

      MS_EXCEPTION_IF_NULL(type_ptr);

      if (!is_ascend) {
        (void)CheckAndConvertUtils::CheckTypeValid("x", type_ptr, target_dtypes, prim_name);
      } else {
        (void)CheckAndConvertUtils::CheckTypeValid("x", type_ptr, default_target_dtypes, prim_name);
      }
      out_types.push_back(type_ptr);
    }
    if (isend_tensor_count != send_type_index) {
      MS_LOG(EXCEPTION) << "BatchISendIRecv input tensors count not match 'isend' in 'op_types'.";
    }
    return std::make_shared<Tuple>(std::move(out_types));
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);

    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    return abstract::MakeAbstract(shape, type);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchISendIRecv, prim::kPrimBatchISendIRecv, BatchISendIRecvInfer, false);
}  // namespace ops
}  // namespace mindspore
