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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_CPU_ATTR_UTILS_OP_ATTR_TYPE_H_
#define MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_CPU_ATTR_UTILS_OP_ATTR_TYPE_H_
namespace mindspore {
namespace predict {
namespace convert {
typedef enum CpuOpType {
  CPU_OP_PAD = 0,
  CPU_OP_MAXIMUM,
  CPU_OP_CONCAT,
  CPU_OP_SOFTMAX,
  CPU_OP_ACTIVATION,
  CPU_OP_CONV2D,
  CPU_OP_FUSEDBATCHNORM,
  CPU_OP_CAFFEBATCHNORM,
  CPU_OP_SQUEEZE,
  CPU_OP_BIASADD,
  CPU_OP_POOLING,
  CPU_OP_DEPTHWISECONV2D,
  CPU_OP_DEDEPTHWISECONV2D,
  CPU_OP_RESIZE,
  CPU_OP_DETECTIONPOSTPROCESS,
  CPU_OP_FULLCONNECTION,
  CPU_OP_MEAN,
  CPU_OP_DECONV2D,
  CPU_OP_SCALE,
  CPU_OP_ELTWISE,
  CPU_OP_ADD,
  CPU_OP_SLICE,
  CPU_OP_MUL,
  CPU_OP_EXP,
  CPU_OP_RESHAPE,
  CPU_OP_POWER,
  CPU_OP_ARGMAX,
  CPU_OP_ARGMAX_NETOUTPUT,
  CPU_OP_MATMUL,
  CPU_OP_CAFFEPRELU,
  CPU_OP_STRIDEDSLICE,
  CPU_OP_STACK,
  CPU_OP_RANGE,
  CPU_OP_EXPANDDIMS,
  CPU_OP_TILE,
  CPU_OP_CAST,
  CPU_OP_CAFFECROP,
  CPU_OP_PRESERVEED = 37
} CpuOpType_t;
}  // namespace convert
}  // namespace predict
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_CPU_ATTR_UTILS_OP_ATTR_TYPE_H_
