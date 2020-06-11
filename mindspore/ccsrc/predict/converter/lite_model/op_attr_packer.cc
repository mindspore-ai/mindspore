/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "predict/converter/lite_model/op_attr_packer.h"
#include "./securec.h"

namespace mindspore {
namespace predict {
namespace convert {
// forward declare
bool Conv2dPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool MatMulPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool BiasAddPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool ReshapePacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool ActivationPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool PoolingPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool FusedBatchNormPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool AddPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool CastPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool MeanPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool SoftmaxPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool ScalePacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool AddFoldPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool ArgMaxPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool BatchNormFoldPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool FakeQuantWithMinMaxPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool FakeQuantWithMinMaxPerChannelPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool MulPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool MulFoldPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);
bool SqueezePacker(const CNodePtr &c_node_ptr, OpDefT *ms_op);

OpAttrFactory::OpAttrFactory() {
  pack_funs_ = {{"Conv2D", Conv2dPacker},
                {"MatMul", MatMulPacker},
                {"BiasAdd", BiasAddPacker},
                {"Reshape", ReshapePacker},
                {"Activation", ActivationPacker},
                {"ReLU", ActivationPacker},
                {"ReLU6", ActivationPacker},
                {"EReLU", ActivationPacker},
                {"LeakyReLU", ActivationPacker},
                {"Sigmoid", ActivationPacker},
                {"Softsign", ActivationPacker},
                {"Softplus", ActivationPacker},
                {"Tanh", ActivationPacker},
                {"HSwish", ActivationPacker},
                {"HSigmoid", ActivationPacker},
                {"MaxPool", PoolingPacker},
                {"MaxPool2D", PoolingPacker},
                {"MeanPool", PoolingPacker},
                {"GlobalPool", PoolingPacker},
                {"FusedBatchNorm", FusedBatchNormPacker},
                {"FusedBatchNormGrad", FusedBatchNormPacker},
                {"Cast", CastPacker},
                {"TensorAdd", AddPacker},
                {"SoftMax", SoftmaxPacker},
                {"SimpleMean", MeanPacker},
                {"ReduceMean", MeanPacker},
                {"AddFold", AddFoldPacker},
                {"ArgMax", ArgMaxPacker},
                {"BatchNorm", BatchNormFoldPacker},
                {"FakeQuantWithMinMax", FakeQuantWithMinMaxPacker},
                {"FakeQuantWithMinMaxPerChannel", FakeQuantWithMinMaxPerChannelPacker},
                {"Mul", MulPacker},
                {"MulFold", MulFoldPacker},
                {"Squeeze", SqueezePacker}};
}
OpAttrPackFun OpAttrFactory::GetPackFun(const std::string &opType) {
  if (pack_funs_.find(opType) == pack_funs_.end()) {
    MS_LOG(WARNING) << "Op Attr pack fun  [" << opType << "] not found.";
    return nullptr;
  }
  return pack_funs_[opType];
}

mindspore::predict::Format GetAttrFormat(const std::string &format) {
  if (format == kOpFormat_NCHW) {
    return predict::Format::Format_NCHW;
  } else if (format == kOpFormat_NHWC) {
    return predict::Format::Format_NHWC;
  } else {
    return predict::Format::Format_NUM_OF_FORMAT;
  }
}

mindspore::predict::PadMode GetAttrPadMode(const std::string &pad_mode) {
  if (pad_mode == "same") {
    return mindspore::predict::PadMode::PadMode_SAME;
  } else if (pad_mode == "valid") {
    return mindspore::predict::PadMode::PadMode_VALID;
  } else {
    return mindspore::predict::PadMode::PadMode_NOTSET;
  }
}
}  // namespace convert
}  // namespace predict
}  // namespace mindspore
