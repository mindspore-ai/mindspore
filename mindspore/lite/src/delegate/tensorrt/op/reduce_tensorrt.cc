/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/tensorrt/op/reduce_tensorrt.h"

namespace mindspore::lite {
int ReduceTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  auto reduce_op = primitive->value_as_ReduceFusion();
  if (reduce_op == nullptr) {
    MS_LOG(ERROR) << "convert failed";
    return RET_ERROR;
  }
  if (in_tensors.size() != 2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
  }
  auto it = reduce_ops_.find(reduce_op->mode());
  if (it != reduce_ops_.end()) {
    reduce_op_ = it->second;
  } else {
    MS_LOG(ERROR) << "unsupported ReduceMode: " << reduce_op->mode();
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  auto reduce_op = op_primitive_->value_as_ReduceFusion();
  if (reduce_op == nullptr) {
    MS_LOG(ERROR) << "convert failed";
    return RET_ERROR;
  }
  bool keep_dims = reduce_op->keep_dims();
  // axis
  uint32_t reduceAxes = 0;
  mindspore::MSTensor axis_tensor = this->in_tensors_[1];
  if (axis_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << "invalid axis_tensor";
    return RET_ERROR;
  }
  if (axis_tensor.DataType() != DataType::kNumberTypeInt32) {
    MS_LOG(WARNING) << "not int data type";
  }
  int *axis_data = reinterpret_cast<int *>(axis_tensor.MutableData());
  for (int i = 0; i < axis_tensor.ElementNum(); i++) {
    reduceAxes |= (16 - (1u << *axis_data));
    axis_data++;
  }
  MS_LOG(INFO) << "reduceAxes: " << reduceAxes;
  nvinfer1::IReduceLayer *layer = network->addReduce(*tensorrt_in_tensors_[0], reduce_op_, reduceAxes, keep_dims);
  if (layer == nullptr) {
    MS_LOG(ERROR) << "addReduce failed for TensorRT.";
    return RET_ERROR;
  }
  layer->setName(op_name_.c_str());

  nvinfer1::ITensor *out_tensor = layer->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "addReduce output tensor create failed for TensorRT.";
    return RET_ERROR;
  }
  out_tensor->setName(out_tensors_[0].Name().c_str());
  this->AddInnerOutTensors(out_tensor);
  return RET_OK;
}
}  // namespace mindspore::lite
