/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_conv_base_parser.h"
#include <vector>
#include <string>
#include "ops/fusion/conv2d_fusion.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore::lite {
namespace {
constexpr size_t kNumDim1 = 1;
constexpr size_t kNumDim2 = 2;
constexpr size_t kNumDim3 = 3;
constexpr size_t kNumDim4 = 4;
constexpr size_t kNumDim6 = 6;
constexpr int kIndex0 = 0;
constexpr int kIndex1 = 1;
constexpr int kIndex2 = 2;
constexpr int kBeginIndex1 = 0;
constexpr int kBeginIndex2 = 1;
constexpr int kEndIndex1 = 2;
constexpr int kEndIndex2 = 3;
constexpr int kBegin3DIndex1 = 0;
constexpr int kBegin3DIndex2 = 1;
constexpr int kBegin3DIndex3 = 2;
constexpr int kEnd3DIndex1 = 3;
constexpr int kEnd3DIndex2 = 4;
constexpr int kEnd3DIndex3 = 5;

}  // namespace
STATUS ParseDilations(std::vector<int64_t> *dilation, int *conv_dims, const onnx::AttributeProto &onnx_node_attr) {
  MS_CHECK_TRUE_RET(dilation != nullptr && conv_dims != nullptr, RET_NULL_PTR);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim1:
      *conv_dims = CONV1D_DIM;
      dilation->push_back(kIndex1);
      dilation->push_back(onnx_node_attr.ints(kIndex0));
      break;
    case kNumDim2:
      dilation->push_back(onnx_node_attr.ints(kIndex0));
      dilation->push_back(onnx_node_attr.ints(kIndex1));
      break;
    case kNumDim3:
      *conv_dims = CONV3D_DIM;
      // The length requirement for strides is 5, and the first two values are used for padding
      dilation->push_back(onnx_node_attr.ints(kIndex0));
      dilation->push_back(onnx_node_attr.ints(kIndex1));
      dilation->push_back(onnx_node_attr.ints(kIndex0));
      dilation->push_back(onnx_node_attr.ints(kIndex1));
      dilation->push_back(onnx_node_attr.ints(kIndex2));
      break;
    default:
      MS_LOG(ERROR) << "dilations size " << onnx_node_attr.ints().size() << " is not 1, 2 or 3!";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS ParseKernels(std::vector<int64_t> *kernels, int *conv_dims, const onnx::AttributeProto &onnx_node_attr) {
  MS_CHECK_TRUE_RET(kernels != nullptr && conv_dims != nullptr, RET_NULL_PTR);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim1:
      *conv_dims = CONV1D_DIM;
      kernels->push_back(kIndex1);
      kernels->push_back(onnx_node_attr.ints(kIndex0));
      break;
    case kNumDim2:
      kernels->push_back(onnx_node_attr.ints(kIndex0));
      kernels->push_back(onnx_node_attr.ints(kIndex1));
      break;
    case kNumDim3:
      *conv_dims = CONV3D_DIM;
      kernels->push_back(onnx_node_attr.ints(kIndex0));
      kernels->push_back(onnx_node_attr.ints(kIndex1));
      kernels->push_back(onnx_node_attr.ints(kIndex2));
      break;
    default:
      MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 1, 2 or 3!";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS ParseKernelShape(std::vector<int64_t> *kernels, int *conv_dims, const onnx::AttributeProto &onnx_node_attr) {
  MS_CHECK_TRUE_RET(kernels != nullptr && conv_dims != nullptr, RET_NULL_PTR);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim1:
      *conv_dims = CONV1D_DIM;
      kernels->push_back(kIndex1);
      kernels->push_back(onnx_node_attr.ints(kIndex0));
      break;
    case kNumDim2:
      kernels->push_back(onnx_node_attr.ints(kIndex0));
      kernels->push_back(onnx_node_attr.ints(kIndex1));
      break;
    case kNumDim3:
      *conv_dims = CONV3D_DIM;
      kernels->push_back(onnx_node_attr.ints(kIndex0));
      kernels->push_back(onnx_node_attr.ints(kIndex1));
      kernels->push_back(onnx_node_attr.ints(kIndex2));
      break;
    default:
      MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 1, 2 or 3!";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS ParsePads(std::vector<int64_t> *pads, int *conv_dims, const onnx::AttributeProto &onnx_node_attr) {
  MS_CHECK_TRUE_RET(pads != nullptr && conv_dims != nullptr, RET_NULL_PTR);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim2:
      *conv_dims = CONV1D_DIM;
      pads->push_back(kIndex0);
      pads->push_back(kIndex0);
      pads->push_back(onnx_node_attr.ints(kIndex0));
      pads->push_back(onnx_node_attr.ints(kIndex1));
      break;
    case kNumDim4:
      pads->push_back(onnx_node_attr.ints(kBeginIndex1));
      pads->push_back(onnx_node_attr.ints(kEndIndex1));
      pads->push_back(onnx_node_attr.ints(kBeginIndex2));
      pads->push_back(onnx_node_attr.ints(kEndIndex2));
      break;
    case kNumDim6:
      *conv_dims = CONV3D_DIM;
      pads->push_back(onnx_node_attr.ints(kBegin3DIndex1));
      pads->push_back(onnx_node_attr.ints(kEnd3DIndex1));
      pads->push_back(onnx_node_attr.ints(kBegin3DIndex2));
      pads->push_back(onnx_node_attr.ints(kEnd3DIndex2));
      pads->push_back(onnx_node_attr.ints(kBegin3DIndex3));
      pads->push_back(onnx_node_attr.ints(kEnd3DIndex3));
      break;
    default:
      MS_LOG(ERROR) << "pads size " << onnx_node_attr.ints().size() << " is not 2, 4 or 6!";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS ParseStrides(std::vector<int64_t> *strides, int *conv_dims, const onnx::AttributeProto &onnx_node_attr) {
  MS_CHECK_TRUE_RET(strides != nullptr && conv_dims != nullptr, RET_NULL_PTR);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim1:
      *conv_dims = CONV1D_DIM;
      strides->push_back(1);
      strides->push_back(onnx_node_attr.ints(kIndex0));
      break;
    case kNumDim2:
      strides->push_back(onnx_node_attr.ints(kIndex0));
      strides->push_back(onnx_node_attr.ints(kIndex1));
      break;
    case kNumDim3:
      *conv_dims = CONV3D_DIM;
      // The length requirement for strides is 5, and the first two values are used for padding
      strides->push_back(onnx_node_attr.ints(kIndex0));
      strides->push_back(onnx_node_attr.ints(kIndex1));
      strides->push_back(onnx_node_attr.ints(kIndex0));
      strides->push_back(onnx_node_attr.ints(kIndex1));
      strides->push_back(onnx_node_attr.ints(kIndex2));
      break;
    default:
      MS_LOG(ERROR) << "strides size " << onnx_node_attr.ints().size() << " is not 1, 2 or 3!";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS OnnxConvBaseParser::ParseVecAttr(const onnx::NodeProto &onnx_node, std::vector<int64_t> *kernels,
                                        std::vector<int64_t> *strides, std::vector<int64_t> *dilation,
                                        std::vector<int64_t> *pads, int *conv_dims) {
  MS_CHECK_TRUE_RET(kernels != nullptr && strides != nullptr && dilation != nullptr, RET_NULL_PTR);
  MS_CHECK_TRUE_RET(pads != nullptr && conv_dims != nullptr, RET_NULL_PTR);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "dilations") {
      auto ret = ParseDilations(dilation, conv_dims, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse dilations failed!";
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "kernels") {
      auto ret = ParseKernels(kernels, conv_dims, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse kernels failed!";
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "kernel_shape") {
      auto ret = ParseKernelShape(kernels, conv_dims, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse kernel_shape failed!";
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "pads") {
      auto ret = ParsePads(pads, conv_dims, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse pads failed!";
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "strides") {
      auto ret = ParseStrides(strides, conv_dims, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse strides failed!";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
