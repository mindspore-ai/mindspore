/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/conv2d.h"
#include <string>
#include <memory>
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#ifdef PRIMITIVE_WRITEABLE
#include "tools/converter/quantizer/quantize_util.h"
#endif

namespace mindspore {
namespace lite {
int Conv2D::PadUp() const { return this->pad_u_; }
int Conv2D::PadDown() const { return this->pad_d_; }
int Conv2D::PadLeft() const { return this->pad_l_; }
int Conv2D::PadRight() const { return this->pad_r_; }
#ifdef PRIMITIVE_WRITEABLE
int Conv2D::GetFormat() const { return this->primitive_->value.AsConv2D()->format; }
int Conv2D::GetGroup() const { return this->primitive_->value.AsConv2D()->group; }
int Conv2D::GetChannelIn() const { return this->primitive_->value.AsConv2D()->channelIn; }
int Conv2D::GetChannelOut() const { return this->primitive_->value.AsConv2D()->channelOut; }
int Conv2D::GetKernelW() const { return this->primitive_->value.AsConv2D()->kernelW; }
int Conv2D::GetKernelH() const { return this->primitive_->value.AsConv2D()->kernelH; }
int Conv2D::GetStrideW() const { return this->primitive_->value.AsConv2D()->strideW; }
int Conv2D::GetStrideH() const { return this->primitive_->value.AsConv2D()->strideH; }
int Conv2D::GetPadMode() const { return this->primitive_->value.AsConv2D()->padMode; }
int Conv2D::GetPadUp() const { return this->primitive_->value.AsConv2D()->padUp; }
int Conv2D::GetPadDown() const { return this->primitive_->value.AsConv2D()->padDown; }
int Conv2D::GetPadLeft() const { return this->primitive_->value.AsConv2D()->padLeft; }
int Conv2D::GetPadRight() const { return this->primitive_->value.AsConv2D()->padRight; }
int Conv2D::GetDilateW() const { return this->primitive_->value.AsConv2D()->dilateW; }
int Conv2D::GetDilateH() const { return this->primitive_->value.AsConv2D()->dilateH; }
bool Conv2D::GetHasBias() const { return this->primitive_->value.AsConv2D()->hasBias; }
int Conv2D::GetActivationType() const { return this->primitive_->value.AsConv2D()->activationType; }

void Conv2D::SetFormat(int format) { this->primitive_->value.AsConv2D()->format = (schema::Format)format; }
void Conv2D::SetGroup(int group) { this->primitive_->value.AsConv2D()->group = group; }
void Conv2D::SetChannelIn(int channel_in) { this->primitive_->value.AsConv2D()->channelIn = channel_in; }
void Conv2D::SetChannelOut(int channel_out) { this->primitive_->value.AsConv2D()->channelOut = channel_out; }
void Conv2D::SetKernelW(int kernel_w) { this->primitive_->value.AsConv2D()->kernelW = kernel_w; }
void Conv2D::SetKernelH(int kernel_h) { this->primitive_->value.AsConv2D()->kernelH = kernel_h; }
void Conv2D::SetStrideW(int stride_w) { this->primitive_->value.AsConv2D()->strideW = stride_w; }
void Conv2D::SetStrideH(int stride_h) { this->primitive_->value.AsConv2D()->strideH = stride_h; }
void Conv2D::SetPadMode(int pad_mode) { this->primitive_->value.AsConv2D()->padMode = (schema::PadMode)pad_mode; }
void Conv2D::SetPadUp(int pad_up) { this->primitive_->value.AsConv2D()->padUp = pad_up; }
void Conv2D::SetPadDown(int pad_down) { this->primitive_->value.AsConv2D()->padDown = pad_down; }
void Conv2D::SetPadLeft(int pad_left) { this->primitive_->value.AsConv2D()->padLeft = pad_left; }
void Conv2D::SetPadRight(int pad_right) { this->primitive_->value.AsConv2D()->padRight = pad_right; }
void Conv2D::SetDilateW(int dilate_w) { this->primitive_->value.AsConv2D()->dilateW = dilate_w; }
void Conv2D::SetDilateH(int dilate_h) { this->primitive_->value.AsConv2D()->dilateH = dilate_h; }
void Conv2D::SetHasBias(bool has_bias) { this->primitive_->value.AsConv2D()->hasBias = has_bias; }
void Conv2D::SetActivationType(int activation_type) {
  this->primitive_->value.AsConv2D()->activationType = (schema::ActivationType)activation_type;
}
template <typename T>
void ConvertConvWeight(const ParameterPtr &param_node) {
  MS_ASSERT(param_node != nullptr);
  auto param = param_node->default_param();
  auto weight = std::dynamic_pointer_cast<ParamValueLite>(param);
  MS_ASSERT(weight != nullptr);

  std::unique_ptr<T> buf(new (std::nothrow) T[weight->tensor_shape_size()]);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "new buf failed";
    return;
  }

  size_t filter_k = weight->tensor_shape()[0];
  size_t filter_c = weight->tensor_shape()[1];
  size_t filter_h = weight->tensor_shape()[2];
  size_t filter_w = weight->tensor_shape()[3];
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (size_t k = 0; k < filter_k; ++k) {
    for (size_t c = 0; c < filter_c; ++c) {
      for (size_t h = 0; h < filter_h; ++h) {
        for (size_t w = 0; w < filter_w; ++w) {
          p1Buff = reinterpret_cast<float *>(weight->tensor_addr()) +
                   ((k * filter_c * filter_h * filter_w) + (c * filter_h * filter_w) + (h * filter_w) + (w));
          p2Buff =
            buf.get() + ((c * filter_k * filter_h * filter_w) + (k * filter_h * filter_w) + (h * filter_w) + (w));
          *p2Buff = *p1Buff;
        }
      }
    }
  }

  auto ret = ::memcpy_s(weight->tensor_addr(), weight->tensor_shape_size() * sizeof(T), buf.get(),
                        weight->tensor_shape_size() * sizeof(T));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed: " << ret;
    return;
  }

  auto abstract_base = param_node->abstract();
  MS_ASSERT(abstract_base != nullptr);
  if (utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
    utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[0] = filter_c;
    utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[1] = filter_k;
    utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[2] = filter_h;
    utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[3] = filter_w;
  }
  return;
}
void Conv2D::PopulaterConv2DMultiGroup(const Primitive &prim, schema::PrimitiveT *primitive, const int &group,
                                       const std::vector<AnfNodePtr> &inputs) {
  auto attr = std::make_unique<schema::DepthwiseConv2DT>();
  auto format = GetValue<std::string>(prim.GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format_NHWC;
  } else {
    attr->format = schema::Format_NUM_OF_FORMAT;
  }
  auto pad_list = GetValue<std::vector<int>>(prim.GetAttr("pad_list"));
  attr->padUp = pad_list[0];
  attr->padDown = pad_list[1];
  attr->padLeft = pad_list[2];
  attr->padRight = pad_list[3];

  auto dilation = GetValue<std::vector<int>>(prim.GetAttr("dilation"));
  attr->dilateH = dilation[0];
  attr->dilateW = dilation[1];

  auto kernel_size = GetValue<std::vector<int>>(prim.GetAttr("kernel_size"));
  attr->kernelH = kernel_size[0];
  attr->kernelW = kernel_size[1];

  auto stride = GetValue<std::vector<int>>(prim.GetAttr("stride"));
  attr->strideH = stride[2];
  attr->strideW = stride[3];

  auto pad_mode = GetValue<std::string>(prim.GetAttr("pad_mode"));
  if (pad_mode == "valid") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "same") {
    attr->padMode = schema::PadMode_SAME;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }

  int channel_mutiplier = 1;
  if (prim.GetAttr("channel_mutiplier") != nullptr) {
    channel_mutiplier = GetValue<int>(prim.GetAttr("channel_multiplier"));
  }
  attr->channelMultiplier = channel_mutiplier;

  MS_ASSERT(inputs.size() == kAnfPopulaterTwo);
  auto input_node = inputs[kAnfPopulaterOne];
  MS_ASSERT(input_node != nullptr);
  if (input_node->isa<Parameter>()) {
    auto param_node = input_node->cast<ParameterPtr>();
    ConvertConvWeight<float>(param_node);
  }

  primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
  primitive->value.value = attr.release();
}

void Conv2D::PopulaterConv2DSingleGroup(const Primitive &prim, schema::PrimitiveT *primitive, const int &group) {
  auto attr = std::make_unique<schema::Conv2DT>();
  attr->group = group;
  auto format = GetValue<std::string>(prim.GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format_NHWC;
  } else {
    attr->format = schema::Format_NUM_OF_FORMAT;
  }
  auto pad_list = GetValue<std::vector<int>>(prim.GetAttr("pad_list"));
  attr->padUp = pad_list[0];
  attr->padDown = pad_list[1];
  attr->padLeft = pad_list[2];
  attr->padRight = pad_list[3];

  auto dilation = GetValue<std::vector<int>>(prim.GetAttr("dilation"));
  attr->dilateH = dilation[0];
  attr->dilateW = dilation[1];

  auto kernel_size = GetValue<std::vector<int>>(prim.GetAttr("kernel_size"));
  attr->kernelH = kernel_size[0];
  attr->kernelW = kernel_size[1];

  auto stride = GetValue<std::vector<int>>(prim.GetAttr("stride"));
  attr->strideH = stride[2];
  attr->strideW = stride[3];

  attr->channelOut = GetValue<int>(prim.GetAttr("out_channel"));

  auto pad_mode = GetValue<std::string>(prim.GetAttr("pad_mode"));
  if (pad_mode == "valid") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "same") {
    attr->padMode = schema::PadMode_SAME;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }
  primitive->value.type = schema::PrimitiveType_Conv2D;
  primitive->value.value = attr.release();
}

void Conv2D::CalQuantParam(const double &mean, const double &stdDev, float *mMin, float *mMax) {
  constexpr float qmin = 0;
  constexpr float qmax = 255;
  *mMin = static_cast<float>((qmin - mean) / stdDev);
  *mMax = static_cast<float>((qmax - mean) / stdDev);
}

void Conv2D::PopulaterQuantParam(const Primitive &prim,
                                 std::vector<std::vector<schema::QuantParamT>> *vecInputQuantParam,
                                 std::vector<std::vector<schema::QuantParamT>> *vecOutputQuantParam) {
  auto narrow_range = prim.GetAttr("narrow_range");
  bool narrowRangeQuantParam = GetValue<bool>(narrow_range);
  auto num_bits = prim.GetAttr("num_bits");
  int32_t numbitsRangeQuantParam = GetValue<int32_t>(num_bits);

  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quantParam;
  auto mean = prim.GetAttr("mean");
  auto std_dev = prim.GetAttr("std_dev");
  if (mean != nullptr && std_dev != nullptr) {
    auto meanQuantOaram = GetValue<double>(mean);
    double stddevQuantOaram = GetValue<double>(std_dev);
    float mMin = 0.0;
    float mMax = 0.0;
    CalQuantParam(meanQuantOaram, stddevQuantOaram, &mMin, &mMax);
    quantParam.min = mMin;
    quantParam.max = mMax;
  } else {
    auto inputMin = prim.GetAttr("input_minq");
    auto inputMax = prim.GetAttr("input_maxq");
    auto inputMinPtr = inputMin->cast<lite::tensor::TensorPtr>();
    auto inputMaxPtr = inputMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(inputMinPtr->Data());
    float *maxBuf = static_cast<float *>(inputMaxPtr->Data());
    quantParam.min = *minBuf;
    quantParam.max = *maxBuf;
  }
  quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max, narrowRangeQuantParam,
                               numbitsRangeQuantParam);
  quants.emplace_back(quantParam);
  vecInputQuantParam->emplace_back(quants);

  quants.clear();
  int biasQuantSize = 0;
  auto filterMin = prim.GetAttr("filter_minq");
  auto filterMax = prim.GetAttr("filter_maxq");
  if (filterMin != nullptr && filterMax != nullptr) {
    auto filterMinPtr = filterMin->cast<lite::tensor::TensorPtr>();
    auto filterMaxPtr = filterMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(filterMinPtr->Data());
    float *maxBuf = static_cast<float *>(filterMaxPtr->Data());
    biasQuantSize = filterMinPtr->DataSize();
    for (int i = 0; i < biasQuantSize; ++i) {
      quantParam.min = *(minBuf++);
      quantParam.max = *(maxBuf++);
      quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max, narrowRangeQuantParam,
                                   numbitsRangeQuantParam);
      quants.emplace_back(quantParam);
    }
    vecInputQuantParam->emplace_back(quants);
  }

  quants.clear();
  for (int i = 0; i < biasQuantSize; ++i) {
    quantParam.min = 0.0;
    quantParam.max = 0.0;
    quantParam.zeroPoint = 0;

    quantParam.scale = vecInputQuantParam->at(0).at(0).scale * vecInputQuantParam->at(1).at(i).scale;
    quants.emplace_back(quantParam);
  }
  vecInputQuantParam->emplace_back(quants);

  quants.clear();
  auto outputMin = prim.GetAttr("output_minq");
  auto outputMax = prim.GetAttr("output_maxq");
  if (outputMin != nullptr && outputMax != nullptr) {
    auto outputMinPtr = outputMin->cast<lite::tensor::TensorPtr>();
    auto outputMaxPtr = outputMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(outputMinPtr->Data());
    float *maxBuf = static_cast<float *>(outputMaxPtr->Data());
    quantParam.min = *minBuf;
    quantParam.max = *maxBuf;
    quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max, narrowRangeQuantParam,
                                 numbitsRangeQuantParam);
    quants.emplace_back(quantParam);
    vecOutputQuantParam->emplace_back(quants);
  }
}

int Conv2D::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Conv2D;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Conv2D) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }
  int group = GetValue<int>(prim.GetAttr("group"));
  if (group > 1) {
    PopulaterConv2DMultiGroup(prim, this->primitive_, group, inputs);
  } else {
    PopulaterConv2DSingleGroup(prim, this->primitive_, group);
  }

  if (GetQuantType() == schema::QuantType_AwareTraining) {
    std::vector<std::vector<schema::QuantParamT>> vecInputQuantParam;
    std::vector<std::vector<schema::QuantParamT>> vecOutputQuantParam;
    PopulaterQuantParam(prim, &vecInputQuantParam, &vecOutputQuantParam);
    SetInputQuantParam(vecInputQuantParam);
    SetOutputQuantParam(vecOutputQuantParam);
  }
  return RET_OK;
}

#else

int Conv2D::GetFormat() const { return this->primitive_->value_as_Conv2D()->format(); }
int Conv2D::GetGroup() const { return this->primitive_->value_as_Conv2D()->group(); }
int Conv2D::GetChannelIn() const { return this->primitive_->value_as_Conv2D()->channelIn(); }
int Conv2D::GetChannelOut() const { return this->primitive_->value_as_Conv2D()->channelOut(); }
int Conv2D::GetKernelW() const { return this->primitive_->value_as_Conv2D()->kernelW(); }
int Conv2D::GetKernelH() const { return this->primitive_->value_as_Conv2D()->kernelH(); }
int Conv2D::GetStrideW() const { return this->primitive_->value_as_Conv2D()->strideW(); }
int Conv2D::GetStrideH() const { return this->primitive_->value_as_Conv2D()->strideH(); }
int Conv2D::GetPadMode() const { return this->primitive_->value_as_Conv2D()->padMode(); }
int Conv2D::GetPadUp() const { return this->primitive_->value_as_Conv2D()->padUp(); }
int Conv2D::GetPadDown() const { return this->primitive_->value_as_Conv2D()->padDown(); }
int Conv2D::GetPadLeft() const { return this->primitive_->value_as_Conv2D()->padLeft(); }
int Conv2D::GetPadRight() const { return this->primitive_->value_as_Conv2D()->padRight(); }
int Conv2D::GetDilateW() const { return this->primitive_->value_as_Conv2D()->dilateW(); }
int Conv2D::GetDilateH() const { return this->primitive_->value_as_Conv2D()->dilateH(); }
bool Conv2D::GetHasBias() const { return this->primitive_->value_as_Conv2D()->hasBias(); }
int Conv2D::GetActivationType() const { return this->primitive_->value_as_Conv2D()->activationType(); }

#endif
void Conv2D::ConvInferShape(int input_h, int input_w, int *output_h, int *output_w) {
  MS_ASSERT(this->primitive_ != nullptr);
  int kernel_w = GetKernelW();
  int kernel_h = GetKernelH();
  int stride_w = GetStrideW();
  int stride_h = GetStrideH();
  int dilate_w = GetDilateW();
  int dilate_h = GetDilateH();
  pad_l_ = GetPadLeft();
  pad_u_ = GetPadUp();
  pad_d_ = GetPadDown();
  pad_r_ = GetPadRight();

  if (GetPadMode() == schema::PadMode_SAME) {
    *output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(stride_w));
    *output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(stride_h));
    auto pad_h_all = ((*output_h - 1) * stride_h + (kernel_h - 1) * dilate_h + 1 - input_h);
    auto pad_w_all = ((*output_w - 1) * stride_w + (kernel_w - 1) * dilate_w + 1 - input_w);
    pad_u_ = pad_h_all / 2;
    pad_d_ = pad_h_all - pad_u_;
    pad_l_ = pad_w_all / 2;
    pad_r_ = pad_w_all - pad_l_;
  } else {
    *output_w = std::ceil((static_cast<float>(input_w) + pad_l_ + pad_r_ -
                           (static_cast<float>(kernel_w) - 1) * static_cast<float>(dilate_w)) /
                          static_cast<float>(stride_w));
    *output_h = std::ceil((static_cast<float>(input_h) + pad_u_ + pad_d_ -
                           (static_cast<float>(kernel_h) - 1) * static_cast<float>(dilate_h)) /
                          static_cast<float>(stride_h));
  }
}

int Conv2D::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  if (inputs_.size() != 2 && inputs_.size() != 3) {
    MS_LOG(ERROR) << "Add should has two or three inputs";
    return RET_ERROR;
  }
  if (outputs_.size() != 1) {
    MS_LOG(ERROR) << "Add should has one outputs";
    return RET_ERROR;
  }
  auto *input_tensor = inputs_.front();
  auto *weight_tensor = inputs_.at(1);
  auto *out_tensor = outputs_.front();
  MS_ASSERT(input_tensor != nullptr);
  MS_ASSERT(out_tensor != nullptr);

  out_tensor->SetFormat(input_tensor->GetFormat());
  out_tensor->set_data_type(input_tensor->data_type());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto in_shape = input_tensor->shape();
  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);
  int output_w = 0, output_h = 0;

  this->ConvInferShape(input_h, input_w, &output_h, &output_w);

  std::vector<int> out_shape{input_tensor->shape()};
  out_shape.at(1) = output_h;
  out_shape.at(2) = output_w;
  out_shape.at(3) = weight_tensor->shape()[0];
  out_tensor->set_shape(out_shape);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
