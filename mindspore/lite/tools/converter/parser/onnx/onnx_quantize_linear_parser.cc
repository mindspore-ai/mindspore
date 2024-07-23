/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_quantize_linear_parser.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <utility>
#include <cstdint>
#include "nnacl/op_base.h"
#include "tools/converter/ops/ops_def.h"
#include "ops/op_utils.h"

namespace mindspore::lite {
namespace {
constexpr size_t kONNXFloat32Type = 1;
constexpr size_t kONNXInt8Type = 3;
}  // namespace
tensor::TensorPtr OnnxQuantizeLinearParser::GetConstData(const onnx::GraphProto &onnx_graph,
                                                         const std::string &input_name) {
  auto node_iter = std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                                [input_name](const onnx::TensorProto &proto) { return proto.name() == input_name; });
  if (node_iter == onnx_graph.initializer().end()) {
    MS_LOG(INFO) << "graph_initializer not find node: " << input_name;
    return nullptr;
  }
  auto tensor = OnnxNodeParser::CopyOnnxTensorData(*node_iter);
  if (tensor == nullptr || tensor->data_c() == nullptr || tensor->Dtype() == nullptr) {
    return nullptr;
  }
  return tensor;
}

template <typename T>
std::vector<T> OnnxQuantizeLinearParser::GetConstTData(const onnx::GraphProto &onnx_graph,
                                                       const std::string &input_name) {
  auto ans_data = OnnxNodeParser::GetConstantTensorData(onnx_graph, input_name);
  if (ans_data == nullptr) {
    MS_LOG(ERROR) << "Failed to find const input from ONNX Constant and initializer, input name: " << input_name;
    return {};
  }
  if (ans_data->data_type() != kONNXFloat32Type && ans_data->data_type() != kONNXInt8Type) {
    MS_LOG(ERROR) << "Only float and int types are supported, but get ONNX data_type: " << ans_data->data_type();
    return {};
  }
  const auto ans_raw_data = reinterpret_cast<const T *>(ans_data->raw_data().data());
  MS_CHECK_TRUE_RET(ans_raw_data != nullptr, {});
  const int64_t ans_size = ans_data->raw_data().size() / sizeof(T);
  std::vector<T> ans_vec;
  ans_vec.resize(ans_size);
  if (INT_MUL_OVERFLOW_THRESHOLD(ans_size, sizeof(T), SIZE_MAX)) {
    MS_LOG(ERROR) << "data_size overflow!";
    return {};
  }
  MS_CHECK_TRUE_RET(ans_vec.data() != nullptr, {});
  if (memcpy_s(ans_vec.data(), ans_size * sizeof(T), ans_raw_data, ans_data->raw_data().size()) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed!";
    return {};
  }
  MS_LOG(INFO) << "ONNX parse success, Constant name: " << input_name << ", size: " << ans_size;
  return ans_vec;
}

bool OnnxQuantizeLinearParser::SetScaleAttr(const onnx::GraphProto &onnx_graph, const string &onnx_quantize_scale,
                                            const std::unique_ptr<QuantizeLinear> &prim) {
  auto onnx_scale_data = GetConstData(onnx_graph, onnx_quantize_scale);
  if (onnx_scale_data == nullptr) {
    std::vector<float> scale_vec = GetConstTData<float>(onnx_graph, onnx_quantize_scale);
    if (scale_vec.empty()) {
      MS_LOG(ERROR) << "scale_vec is empty!";
      return false;
    }
    prim->AddAttr(kAttrScaleVec, MakeValue(scale_vec));
  } else {
    float scale_f = 1;
    scale_f = *(static_cast<const float *>(onnx_scale_data->data_c()));
    prim->AddAttr(kAttrScale, MakeValue(scale_f));
  }
  return true;
}

bool OnnxQuantizeLinearParser::SetZeroPointAttr(const onnx::GraphProto &onnx_graph,
                                                const string &onnx_quantize_zero_point,
                                                const std::unique_ptr<QuantizeLinear> &prim) {
  auto onnx_zero_point_data = GetConstData(onnx_graph, onnx_quantize_zero_point);
  if (onnx_zero_point_data == nullptr) {
    std::vector<int8_t> point_vec = GetConstTData<int8_t>(onnx_graph, onnx_quantize_zero_point);
    if (point_vec.empty()) {
      MS_LOG(ERROR) << "point_vec is empty!";
      return false;
    }
    prim->AddAttr(kAttrZeroPointVec, MakeValue(point_vec));
  } else {
    TypeId zp_data_type = onnx_zero_point_data->Dtype()->type_id();
    void *zp_data = onnx_zero_point_data->data_c();
    MS_CHECK_TRUE_RET(zp_data != nullptr, false);
    int zero_point = 0;
    if (zp_data_type == mindspore::kNumberTypeUInt8) {
      zero_point = *(static_cast<const uint8_t *>(zp_data)) - 128;
    } else if (zp_data_type == mindspore::kNumberTypeInt8) {
      auto zero_point_int8 = *(static_cast<const int8_t *>(zp_data));
      zero_point = static_cast<const int32_t>(zero_point_int8);
    } else {
      MS_LOG(ERROR) << "Invalid zero point data type: " << zp_data_type << ", zero_point is " << zero_point;
      prim->AddAttr(kAttrZeroPoint, MakeValue(zero_point));
      return false;
    }
    prim->AddAttr(kAttrZeroPoint, MakeValue(zero_point));
  }
  return true;
}

PrimitiveCPtr OnnxQuantizeLinearParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<QuantizeLinear>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  MS_CHECK_GE(onnx_node.input_size(), kInputSize2, nullptr);
  const auto &onnx_quantize_scale = onnx_node.input(SECOND_INPUT);
  const auto &onnx_quantize_zero_point = onnx_node.input(THIRD_INPUT);
  if (!SetScaleAttr(onnx_graph, onnx_quantize_scale, prim)) {
    return nullptr;
  }
  if (!SetZeroPointAttr(onnx_graph, onnx_quantize_zero_point, prim)) {
    return nullptr;
  }
  return prim;
}

OnnxNodeRegistrar g_onnxQuantizeLinearParser("QuantizeLinear", new OnnxQuantizeLinearParser());
}  // namespace mindspore::lite
