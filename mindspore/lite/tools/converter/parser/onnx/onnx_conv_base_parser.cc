/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

namespace mindspore::lite {
namespace {
constexpr size_t kNumDim1 = 1;
constexpr size_t kNumDim2 = 2;
constexpr size_t kNumDim3 = 3;
constexpr size_t kNumDim4 = 4;
constexpr int kBeginIndex1 = 0;
constexpr int kBeginIndex2 = 1;
constexpr int kEndIndex1 = 2;
constexpr int kEndIndex2 = 3;
}  // namespace
STATUS ParseDilations(std::vector<int64_t> *dilation, bool *conv1d, const onnx::AttributeProto &onnx_node_attr) {
  MS_ASSERT(dilation != nullptr && conv1d != nullptr);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim1:
      *conv1d = true;
      dilation->push_back(1);
      dilation->push_back(onnx_node_attr.ints(0));
      break;
    case kNumDim2:
      dilation->push_back(onnx_node_attr.ints(0));
      dilation->push_back(onnx_node_attr.ints(1));
      break;
    default:
      MS_LOG(ERROR) << "dilations size " << onnx_node_attr.ints().size() << " is not 1 or 2";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS ParseKernels(std::vector<int64_t> *kernels, bool *conv1d, const onnx::AttributeProto &onnx_node_attr) {
  MS_ASSERT(kernels != nullptr && conv1d != nullptr);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim1:
      *conv1d = true;
      kernels->push_back(1);
      kernels->push_back(onnx_node_attr.ints(0));
      break;
    case kNumDim2:
      kernels->push_back(onnx_node_attr.ints(0));
      kernels->push_back(onnx_node_attr.ints(1));
      break;
    default:
      MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 1 or 2";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS ParseKernelShape(std::vector<int64_t> *kernels, bool *conv1d, const onnx::AttributeProto &onnx_node_attr) {
  MS_ASSERT(kernels != nullptr && conv1d != nullptr);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim1:
      *conv1d = true;
      kernels->push_back(1);
      kernels->push_back(onnx_node_attr.ints(0));
      break;
    case kNumDim2:
      kernels->push_back(onnx_node_attr.ints(0));
      kernels->push_back(onnx_node_attr.ints(1));
      break;
    default:
      MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 1 or 2";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS ParsePads(std::vector<int64_t> *pads, bool *conv1d, const onnx::AttributeProto &onnx_node_attr) {
  MS_ASSERT(pads != nullptr && conv1d != nullptr);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim2:
      *conv1d = true;
      pads->push_back(0);
      pads->push_back(0);
      pads->push_back(onnx_node_attr.ints(0));
      pads->push_back(onnx_node_attr.ints(1));
      break;
    case kNumDim4:
      pads->push_back(onnx_node_attr.ints(kBeginIndex1));
      pads->push_back(onnx_node_attr.ints(kEndIndex1));
      pads->push_back(onnx_node_attr.ints(kBeginIndex2));
      pads->push_back(onnx_node_attr.ints(kEndIndex2));
      break;
    default:
      MS_LOG(ERROR) << "pads size " << onnx_node_attr.ints().size() << " is not 2 or 4";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS ParseStrides(std::vector<int64_t> *strides, bool *conv1d, const onnx::AttributeProto &onnx_node_attr) {
  MS_ASSERT(strides != nullptr && conv1d != nullptr);
  switch (onnx_node_attr.ints().size()) {
    case kNumDim1:
      *conv1d = true;
      strides->push_back(1);
      strides->push_back(onnx_node_attr.ints(0));
      break;
    case kNumDim2:
      strides->push_back(onnx_node_attr.ints(0));
      strides->push_back(onnx_node_attr.ints(1));
      break;
    default:
      MS_LOG(ERROR) << "strides size " << onnx_node_attr.ints().size() << " is not 1 or 2";
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS OnnxConvBaseParser::ParseVecAttr(const onnx::NodeProto &onnx_node, std::vector<int64_t> *kernels,
                                        std::vector<int64_t> *strides, std::vector<int64_t> *dilation,
                                        std::vector<int64_t> *pads, bool *conv1d) {
  MS_ASSERT(kernels != nullptr);
  MS_ASSERT(strides != nullptr);
  MS_ASSERT(dilation != nullptr);
  MS_ASSERT(pads != nullptr);
  MS_ASSERT(conv1d != nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "dilations") {
      auto ret = ParseDilations(dilation, conv1d, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse dilations failed!";
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "kernels") {
      auto ret = ParseKernels(kernels, conv1d, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse kernels failed!";
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "kernel_shape") {
      auto ret = ParseKernelShape(kernels, conv1d, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse kernel_shape failed!";
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "pads") {
      auto ret = ParsePads(pads, conv1d, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse pads failed!";
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "strides") {
      auto ret = ParseStrides(strides, conv1d, onnx_node_attr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Parse strides failed!";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
