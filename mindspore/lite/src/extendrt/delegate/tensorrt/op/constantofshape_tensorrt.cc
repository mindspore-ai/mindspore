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

#include "src/extendrt/delegate/tensorrt/op/constantofshape_tensorrt.h"
#include <numeric>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/constant_of_shape.h"

namespace mindspore::lite {
int ConstantOfShapeTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                       const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }

  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }
  auto constofshape_op = AsOps<ops::ConstantOfShape>();
  if (constofshape_op == nullptr) {
    MS_LOG(ERROR) << "Failed to as operator ConstantOfShape: " << op_name_;
    return RET_ERROR;
  }
  DataType data_type = static_cast<DataType>(constofshape_op->get_data_type());
  if (data_type != DataType::kNumberTypeInt32 && data_type != DataType::kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Unsupported data type for " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ConstantOfShapeTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto constofshape_op = AsOps<ops::ConstantOfShape>();
  if (constofshape_op == nullptr) {
    MS_LOG(ERROR) << "Failed to as operator ConstantOfShape: " << op_name_;
    return RET_ERROR;
  }
  auto &&value_vector = constofshape_op->get_value();
  auto value_tensor = ctx->ConvertTo1DTensor(value_vector);
  CHECK_NULL_RETURN(value_tensor);

  auto unsqueeze_layer = ctx->network()->addShuffle(*value_tensor);
  CHECK_NULL_RETURN(unsqueeze_layer);

  auto shape = input(ctx, 0).trt_tensor_;
  int rank = shape->getDimensions().d[0];
  nvinfer1::Dims unsqueeze{rank};
  std::fill(unsqueeze.d, unsqueeze.d + rank, 1);
  unsqueeze_layer->setReshapeDimensions(unsqueeze);
  unsqueeze_layer->setZeroIsPlaceholder(false);
  value_tensor = unsqueeze_layer->getOutput(0);
  CHECK_NULL_RETURN(value_tensor);

  auto out_tensor = Broadcast(ctx, value_tensor, shape);
  if (static_cast<DataType>(constofshape_op->get_data_type()) == DataType::kNumberTypeInt32) {
    out_tensor = TRTTensorCast(ctx, out_tensor, nvinfer1::DataType::kINT32, op_name_ + "_cast_out");
  }

  auto output_helper = ITensorHelper{out_tensor, Format::NCHW, true};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  this->layer_ = unsqueeze_layer;
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameConstantOfShape, ConstantOfShapeTensorRT)
}  // namespace mindspore::lite
