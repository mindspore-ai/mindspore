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

#include "src/ops/primitive_c.h"
#ifdef PRIMITIVE_WRITEABLE
#include <memory>
#include <map>

#include "tools/converter/quantizer/quantize_util.h"
#include "src/ops/assert_op.h"
#include "src/ops/space_to_batch.h"
#include "src/ops/space_to_batch_nd.h"
#include "src/ops/conv2d.h"
#include "src/ops/roi_pooling.h"
#include "src/ops/topk.h"
#include "src/ops/broadcast_to.h"
#include "src/ops/unsqueeze.h"
#include "src/ops/unstack.h"
#include "src/ops/depth_to_space.h"
#include "src/ops/batch_to_space.h"
#include "src/ops/prior_box.h"
#include "src/ops/lstm.h"
#include "src/ops/softmax.h"
#include "src/ops/activation.h"
#include "src/ops/deconv2d.h"
#include "src/ops/reduce.h"
#include "src/ops/pooling.h"
#include "src/ops/fused_batchnorm.h"
#include "src/ops/batch_norm.h"
#include "src/ops/power.h"
#include "src/ops/range.h"
#include "src/ops/add.h"
#include "src/ops/sub.h"
#include "src/ops/div.h"
#include "src/ops/bias_add.h"
#include "src/ops/expand_dims.h"
#include "src/ops/full_connection.h"
#include "src/ops/shape.h"
#include "src/ops/elu.h"
#include "src/ops/embedding_lookup.h"
#include "src/ops/quant_dtype_cast.h"
#include "src/ops/matmul.h"
#include "src/ops/resize.h"
#include "src/ops/tile.h"
#include "src/ops/one_hot.h"
#include "src/ops/space_to_depth.h"
#include "src/ops/split.h"
#include "src/ops/argmax.h"
#include "src/ops/argmin.h"
#include "src/ops/cast.h"
#include "src/ops/reshape.h"
#include "src/ops/scale.h"
#include "src/ops/concat.h"
#include "src/ops/nchw2nhwc.h"
#include "src/ops/slice.h"
#include "src/ops/squeeze.h"
#include "src/ops/flatten.h"
#include "src/ops/nhwc2nchw.h"
#include "src/ops/stack.h"
#include "src/ops/crop.h"
#include "src/ops/addn.h"
#include "src/ops/gather.h"
#include "src/ops/gather_nd.h"
#include "src/ops/local_response_normalization.h"
#include "src/ops/pad.h"
#include "src/ops/p_relu.h"
#include "src/ops/leaky_relu.h"
#include "src/ops/reverse_sequence.h"
#include "src/ops/dedepthwise_conv2d.h"
#include "src/ops/depthwise_conv2d.h"
#include "src/ops/mul.h"
#include "src/ops/eltwise.h"
#include "src/ops/fill.h"
#include "src/ops/transpose.h"
#include "src/ops/log.h"
#include "src/ops/abs.h"
#include "src/ops/sin.h"
#include "src/ops/cos.h"
#include "src/ops/sqrt.h"
#include "src/ops/square.h"
#include "src/ops/exp.h"
#include "src/ops/rsqrt.h"
#include "src/ops/maximum.h"
#include "src/ops/minimum.h"
#include "src/ops/strided_slice.h"
#include "src/ops/reverse.h"
#include "src/ops/logical_and.h"
#include "src/ops/logical_or.h"
#include "src/ops/logical_not.h"
#include "src/ops/floor_div.h"
#include "src/ops/floor_mod.h"
#include "src/ops/mod.h"
#include "src/ops/equal.h"
#include "src/ops/not_equal.h"
#include "src/ops/less.h"
#include "src/ops/less_equal.h"
#include "src/ops/greater_equal.h"
#include "src/ops/greater.h"
#include "src/ops/floor.h"
#include "src/ops/squared_difference.h"
#include "src/ops/ceil.h"
#include "src/ops/round.h"
#include "src/ops/unique.h"
#include "src/ops/zeros_like.h"
#include "src/ops/return.h"
#include "src/ops/where.h"
#include "src/ops/scatter_nd.h"
#include "src/ops/constant_of_shape.h"
#include "src/ops/dequant.h"
#include "src/ops/make_tuple.h"
#include "src/ops/quant.h"
#include "src/ops/tuple_get_item.h"
#include "src/ops/l2_norm.h"
#include "src/ops/neg.h"
#include "src/ops/sparse_to_dense.h"
#include "src/ops/detection_post_process.h"
#include "src/ops/dropout.h"
#include "src/ops/real_div.h"
#include "src/ops/lsh_projection.h"
#include "src/ops/hashtable_lookup.h"
#include "src/ops/skip_gram.h"
#include "src/ops/clip.h"
#include "src/ops/adder.h"
#include "src/ops/custom_predict.h"
#include "src/ops/custom_normalize.h"
#include "src/ops/custom_extract_features.h"
#include "src/ops/upsample.h"
#include "src/ops/layer_norm.h"
#include "src/ops/non_max_suppression.h"
#include "src/ops/rfft.h"
#include "src/ops/fft_real.h"
#include "src/ops/fft_imag.h"
#include "src/ops/audio_spectrogram.h"
#include "src/ops/mfcc.h"
#include "src/ops/identity.h"
#include "src/ops/instance_norm.h"
#include "src/ops/while.h"
#include "src/ops/oneslike.h"
#include "src/ops/unsorted_segment_sum.h"
#include "src/ops/reciprocal.h"
#include "src/ops/constant.h"
#include "src/ops/tensorlist_fromtensor.h"
#include "src/ops/tensorlist_getitem.h"
#include "src/ops/tensorlist_setitem.h"
#include "src/ops/tensorlist_reserve.h"
#include "src/ops/tensorlist_stack.h"
#include "src/ops/merge.h"
#include "src/ops/switch.h"
#include "src/ops/partial.h"
#include "src/ops/if.h"
#include "src/ops/select.h"
#include "src/ops/gelu.h"
#include "src/ops/gru.h"
#include "src/ops/size.h"
#include "src/ops/random_standard_normal.h"
#include "src/ops/invert_permutation.h"
#include "src/ops/crop_and_resize.h"
#include "src/ops/nonzero.h"

#ifdef SUPPORT_TRAIN
#include "src/ops/neg_grad.h"
#include "src/ops/activation_grad.h"
#include "src/ops/apply_momentum.h"
#include "src/ops/bias_grad.h"
#include "src/ops/pooling_grad.h"
#include "src/ops/conv2d_grad_filter.h"
#include "src/ops/conv2d_grad_input.h"
#include "src/ops/group_conv2d_grad_input.h"
#include "src/ops/power_grad.h"
#include "src/ops/softmax_cross_entropy.h"
#include "src/ops/sparse_softmax_cross_entropy.h"
#include "src/ops/bn_grad.h"
#include "src/ops/arithmetic_grad.h"
#include "src/ops/depend.h"
#include "src/ops/flatten_grad.h"
#include "src/ops/log_grad.h"
#include "src/ops/sgd.h"
#include "src/ops/adam.h"
#include "src/ops/assign.h"
#include "src/ops/dropout_grad.h"
#include "src/ops/maximum_grad.h"
#include "src/ops/minimum_grad.h"
#include "src/ops/control_depend.h"
#include "src/ops/assign_add.h"
#include "src/ops/binary_cross_entropy.h"
#include "src/ops/binary_cross_entropy_grad.h"
#include "src/ops/smooth_l1_loss.h"
#include "src/ops/smooth_l1_loss_grad.h"
#include "src/ops/sigmoid_cross_entropy_with_logits.h"
#include "src/ops/sigmoid_cross_entropy_with_logits_grad.h"
#endif
#endif
namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> CastToInt(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "valueptr is nullptr.";
    return {};
  }
  std::vector<int> cur_value;
  if (utils::isa<ValueSequeuePtr>(value)) {
    if (value->cast<ValueSequeuePtr>()->value().front()->type()->number_type() == kNumberTypeInt64) {
      auto origin_value = GetValue<std::vector<int64_t>>(value);
      for (size_t index = 0; index < origin_value.size(); ++index) {
        cur_value.push_back(static_cast<int>(origin_value[index]));
      }
    } else {
      cur_value = GetValue<std::vector<int>>(value);
    }
  } else {
    if (value->type()->number_type() == kNumberTypeInt64) {
      cur_value.push_back(static_cast<int>(GetValue<int64_t>(value)));
    } else {
      cur_value.push_back(GetValue<int>(value));
    }
  }
  return cur_value;
}

void PrimitiveC::CalFloatScopeByMeanAndStddev(const double &mean, const double &stdDev, float *mMin, float *mMax) {
  const float qmin = 0;
  const float qmax = 255;
  *mMin = static_cast<float>((qmin - mean) / stdDev);
  *mMax = static_cast<float>((qmax - mean) / stdDev);
}

void PrimitiveC::FillDefaultInputQuantParamIfNeed(const size_t &inputSize) {
  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quantParam;

  if (input_quant_param_.size() == kDoubleNum) {
    quants.clear();
    quantParam.min = 0.0;
    quantParam.max = 0.0;
    quantParam.zeroPoint = 0;
    quantParam.scale = input_quant_param_.at(0).at(0).scale * input_quant_param_.at(1).at(0).scale;
    quants.emplace_back(quantParam);
    input_quant_param_.emplace_back(quants);
  }
  // fill input_quant_param_ by not inited quant_parm
  if (input_quant_param_.size() < inputSize) {
    schema::QuantParamT tmpQuantParam;
    quants.emplace_back(tmpQuantParam);
    input_quant_param_.insert(input_quant_param_.end(), inputSize - input_quant_param_.size(), quants);
  }
}

void PrimitiveC::PopulaterInputQuantParam(const Primitive &prim, const std::vector<AnfNodePtr> &inputs,
                                          bool narrowRangeQuantParam, int32_t numbitsRangeQuantParam) {
  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quantParam;
  auto inputMin = prim.GetAttr("input_minq");
  auto inputMax = prim.GetAttr("input_maxq");
  if (inputMin != nullptr && inputMax != nullptr) {
    auto inputMinPtr = inputMin->cast<TensorPtr>();
    auto inputMaxPtr = inputMax->cast<TensorPtr>();
    auto *minBuf = static_cast<float *>(inputMinPtr->data_c());
    auto *maxBuf = static_cast<float *>(inputMaxPtr->data_c());
    quantParam.min = *minBuf;
    quantParam.max = *maxBuf;
    auto ret = quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max, narrowRangeQuantParam,
                                            numbitsRangeQuantParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Can't calculate quant parameters";
      return;
    }
    quants.emplace_back(quantParam);
    input_quant_param_.emplace_back(quants);
  }

  quants.clear();
  auto filterMin = prim.GetAttr("filter_minq");
  auto filterMax = prim.GetAttr("filter_maxq");
  if (filterMin != nullptr && filterMax != nullptr) {
    auto filterMinPtr = filterMin->cast<TensorPtr>();
    auto filterMaxPtr = filterMax->cast<TensorPtr>();
    auto *minBuf = static_cast<float *>(filterMinPtr->data_c());
    auto *maxBuf = static_cast<float *>(filterMaxPtr->data_c());
    quantParam.min = FLT_MAX;
    quantParam.max = FLT_MIN;
    for (int i = 0; i < filterMinPtr->ElementsNum(); ++i) {
      quantParam.min = (*(minBuf) < quantParam.min) ? (*minBuf) : quantParam.min;
      quantParam.max = (*(maxBuf) > quantParam.max) ? (*maxBuf) : quantParam.max;
      minBuf++;
      maxBuf++;
    }
    auto ret = quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max, true, numbitsRangeQuantParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Can't calculate quant parameters";
      return;
    }
    quants.emplace_back(quantParam);
    input_quant_param_.emplace_back(quants);
  }
  FillDefaultInputQuantParamIfNeed(inputs.size());
}

void PrimitiveC::PopulaterOutputQuantParam(const Primitive &prim, bool narrowRangeQuantParam,
                                           int32_t numbitsRangeQuantParam) {
  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quantParam;
  auto outputMin = prim.GetAttr("output_minq");
  auto outputMax = prim.GetAttr("output_maxq");
  if (outputMin != nullptr && outputMax != nullptr) {
    auto outputMinPtr = outputMin->cast<TensorPtr>();
    auto outputMaxPtr = outputMax->cast<TensorPtr>();
    auto *minBuf = static_cast<float *>(outputMinPtr->data_c());
    auto *maxBuf = static_cast<float *>(outputMaxPtr->data_c());
    quantParam.min = *minBuf;
    quantParam.max = *maxBuf;
    auto ret = quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max, narrowRangeQuantParam,
                                            numbitsRangeQuantParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Can't calculate quant parameters";
      return;
    }
    quants.emplace_back(quantParam);
    output_quant_param_.emplace_back(quants);
  } else {
    schema::QuantParamT tmpQuantParam;
    quants.emplace_back(tmpQuantParam);
    output_quant_param_.emplace_back(quants);
  }
}

void PrimitiveC::PopulaterQuantParam(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  auto narrow_range = prim.GetAttr("narrow_range");
  bool narrowRangeQuantParam = false;
  if (narrow_range != nullptr) {
    if (utils::isa<tensor::TensorPtr>(narrow_range)) {
      auto narrow_range_tensor = narrow_range->cast<tensor::TensorPtr>();
      narrowRangeQuantParam = *reinterpret_cast<bool *>(narrow_range_tensor->data_c());
    } else if (utils::isa<ImmTraits<bool>::type>(narrow_range)) {
      narrowRangeQuantParam = GetValue<bool>(narrow_range);
    } else {
      MS_LOG(ERROR) << "valueptr is invalid.";
      return;
    }
  }
  auto num_bits = prim.GetAttr("num_bits");
  int32_t numbitsRangeQuantParam = 8;
  if (num_bits != nullptr) {
    if (utils::isa<tensor::TensorPtr>(num_bits)) {
      auto num_bits_tensor = num_bits->cast<tensor::TensorPtr>();
      numbitsRangeQuantParam = *reinterpret_cast<int64_t *>(num_bits_tensor->data_c());
    } else if (utils::isa<ImmTraits<int64_t>::type>(num_bits)) {
      numbitsRangeQuantParam = GetValue<int64_t>(num_bits);
    }
  }
  PopulaterInputQuantParam(prim, inputs, narrowRangeQuantParam, numbitsRangeQuantParam);
  PopulaterOutputQuantParam(prim, narrowRangeQuantParam, numbitsRangeQuantParam);
}

void PrimitiveC::GetAttrDataFromInput(const AnfNodePtr &inputNode, std::vector<int> *data) {
  if (inputNode->isa<ValueNode>()) {
    auto valNode = inputNode->cast<ValueNodePtr>();
    MS_ASSERT(valNode != nullptr);
    auto val = valNode->value();
    MS_ASSERT(val != nullptr);
    if (val->isa<ValueTuple>()) {
      auto tuple = val->cast<ValueTuplePtr>();
      MS_ASSERT(tuple != nullptr);
      for (size_t i = 0; i < tuple->size(); i++) {
        auto elem = tuple->value().at(i);
        MS_ASSERT(elem != nullptr);
        data->emplace_back(CastToInt(elem).front());
      }
    }
  }
}

schema::PrimitiveT *PrimitiveC::primitiveT() const { return this->primitive_; }

void PrimitiveC::ClearPrimitiveT() { this->primitive_ = nullptr; }

void PrimitiveC::set_input_quant_params(const std::vector<std::vector<schema::QuantParamT>> &input_quant_param) {
  this->input_quant_param_ = input_quant_param;
}

void PrimitiveC::set_input_quant_param(const size_t &index, const std::vector<schema::QuantParamT> &input_quant_param) {
  if (index >= this->input_quant_param_.size()) {
    this->input_quant_param_.resize(index + 1);
  }
  this->input_quant_param_.at(index) = input_quant_param;
}

void PrimitiveC::set_output_quant_params(const std::vector<std::vector<schema::QuantParamT>> &output_quant_param) {
  this->output_quant_param_ = output_quant_param;
}

void PrimitiveC::set_output_quant_param(const size_t &index,
                                        const std::vector<schema::QuantParamT> &output_quant_param) {
  MS_ASSERT(index < this->output_quant_param_.size());
  this->output_quant_param_.at(index) = output_quant_param;
}

bool PrimitiveC::IsInputQuantParamsInited() {
  if (this->input_quant_param_.empty()) {
    return false;
  }
  for (auto &quant_param : this->input_quant_param_) {
    if (!quant_param.front().inited) {
      return false;
    }
  }
  return true;
}

bool PrimitiveC::IsOutputQuantParamsInited() {
  if (this->output_quant_param_.empty()) {
    return false;
  }
  for (auto &quant_param : this->output_quant_param_) {
    if (!quant_param.front().inited) {
      return false;
    }
  }
  return true;
}

void PrimitiveC::ClearInputOutputQuantParam() {
  input_quant_param_.clear();
  output_quant_param_.clear();
}

void PrimitiveC::AddInputQuantParam(const std::vector<schema::QuantParamT> &quant_param) {
  this->input_quant_param_.emplace_back(quant_param);
}
std::vector<std::vector<schema::QuantParamT>> PrimitiveC::input_quant_params() const { return input_quant_param_; }

void PrimitiveC::AddOutputQuantParam(const std::vector<schema::QuantParamT> &quant_param) {
  this->output_quant_param_.emplace_back(quant_param);
}
std::vector<std::vector<schema::QuantParamT>> PrimitiveC::output_quant_params() const { return output_quant_param_; }

void PrimitiveC::set_quant_type(const schema::QuantType &quant_type) { this->quant_type_ = quant_type; }

schema::QuantType PrimitiveC::quant_type() const { return quant_type_; }

bool PrimitiveC::IsEnableHuffmanCode() const { return enableHuffmanCode; }

void PrimitiveC::SetEnableHuffmanCode(bool enableHuffmanCode) { this->enableHuffmanCode = enableHuffmanCode; }

std::shared_ptr<PrimitiveC> GetReturnPrim() {
  auto return_primitiveT = new (std::nothrow) schema::PrimitiveT;
  if (return_primitiveT == nullptr) {
    MS_LOG(ERROR) << "new PrimitiveT failed";
    return nullptr;
  }
  return_primitiveT->value.type = schema::PrimitiveType_Return;
  return_primitiveT->value.value = new (std::nothrow) schema::ReturnT;
  if (return_primitiveT->value.value == nullptr) {
    MS_LOG(ERROR) << "new ReturnT failed";
    delete (return_primitiveT);
    return nullptr;
  }
  return std::make_shared<Return>(return_primitiveT);
}

std::shared_ptr<PrimitiveC> GetMakeTuplePrim() {
  auto make_tuple_primitiveT = new (std::nothrow) schema::PrimitiveT;
  if (make_tuple_primitiveT == nullptr) {
    MS_LOG(ERROR) << "new PrimitiveT failed";
    return nullptr;
  }
  make_tuple_primitiveT->value.type = schema::PrimitiveType_MakeTuple;
  make_tuple_primitiveT->value.value = new (std::nothrow) schema::MakeTupleT;
  if (make_tuple_primitiveT->value.value == nullptr) {
    MS_LOG(ERROR) << "new MakeTupleT failed";
    delete (make_tuple_primitiveT);
    return nullptr;
  }
  return std::make_shared<MakeTuple>(make_tuple_primitiveT);
}

std::shared_ptr<PrimitiveC> GetTupleGetItemPrim() {
  auto tuple_get_item_primitiveT = new (std::nothrow) schema::PrimitiveT();
  if (tuple_get_item_primitiveT == nullptr) {
    MS_LOG(ERROR) << "new PrimitiveT failed";
    return nullptr;
  }
  tuple_get_item_primitiveT->value.type = schema::PrimitiveType_TupleGetItem;
  tuple_get_item_primitiveT->value.value = new (std::nothrow) schema::TupleGetItemT;
  if (tuple_get_item_primitiveT->value.value == nullptr) {
    MS_LOG(ERROR) << "new TupleGetItemT failed";
    delete (tuple_get_item_primitiveT);
    return nullptr;
  }
  return std::make_shared<TupleGetItem>(tuple_get_item_primitiveT);
}

template <typename T, typename = std::enable_if<std::is_base_of<PrimitiveC, T>::value>>
std::shared_ptr<PrimitiveC> NewPrimitiveC(const mindspore::Primitive &prim, const std::vector<AnfNodePtr> &inputs,
                                          const schema::QuantType &quantType) {
  auto primc = std::make_shared<T>();
  if (primc == nullptr) {
    MS_LOG(ERROR) << "make_shared PrimitiveC failed";
    return nullptr;
  }
  primc->set_quant_type(quantType);
  auto ret = primc->UnPackAttr(prim, inputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnPackAttr failed";
    return nullptr;
  }
  return primc;
}

std::shared_ptr<PrimitiveC> PrimitiveC::Create(const Primitive &prim, const std::vector<AnfNodePtr> &inputs,
                                               const schema::QuantType &quantType) {
  const auto &op_type = prim.name();
  if (op_type == "ReLU" || op_type == "ReLU6" || op_type == "Sigmoid" || op_type == "HSwish" || op_type == "HSigmoid") {
    return NewPrimitiveC<Activation>(prim, inputs, quantType);
  } else if (op_type == "AddN") {
    return NewPrimitiveC<AddN>(prim, inputs, quantType);
  } else if (op_type == "BatchNorm") {
    return NewPrimitiveC<BatchNorm>(prim, inputs, quantType);
  } else if (op_type == "BiasAdd") {
    return NewPrimitiveC<BiasAdd>(prim, inputs, quantType);
  } else if (op_type == "Concat") {
    return NewPrimitiveC<Concat>(prim, inputs, quantType);
  } else if (op_type == "Conv2D") {
    return NewPrimitiveC<Conv2D>(prim, inputs, quantType);
  } else if (op_type == "DepthwiseConv2dNative" || op_type == "DepthwiseConv2D") {
    return NewPrimitiveC<DepthwiseConv2D>(prim, inputs, quantType);
  } else if (op_type == "Dequant") {
    return NewPrimitiveC<Dequant>(prim, inputs, quantType);
  } else if (op_type == "Flatten") {
    return NewPrimitiveC<Flatten>(prim, inputs, quantType);
  } else if ((op_type == "FusedBatchNorm") || (op_type == "FusedBatchNormEx")) {
    return NewPrimitiveC<FusedBatchNorm>(prim, inputs, quantType);
  } else if (op_type == "make_tuple") {
    return NewPrimitiveC<MakeTuple>(prim, inputs, quantType);
  } else if (op_type == "MatMul" || op_type == "BatchMatMul") {
    return NewPrimitiveC<MatMul>(prim, inputs, quantType);
  } else if (op_type == "Mul") {
    return NewPrimitiveC<Mul>(prim, inputs, quantType);
  } else if (op_type == "MaxPool" || op_type == "AvgPool") {
    return NewPrimitiveC<Pooling>(prim, inputs, quantType);
  } else if (op_type == "Quant") {
    return NewPrimitiveC<Quant>(prim, inputs, quantType);
  } else if (op_type == "RealDiv") {
    return NewPrimitiveC<RealDiv>(prim, inputs, quantType);
  } else if (op_type == "ReduceMax") {
    return NewPrimitiveC<Reduce>(prim, inputs, quantType);
  } else if (op_type == "ReduceMean") {
    return NewPrimitiveC<Reduce>(prim, inputs, quantType);
  } else if (op_type == "ReduceMin") {
    return NewPrimitiveC<Reduce>(prim, inputs, quantType);
  } else if (op_type == "ReduceProd") {
    return NewPrimitiveC<Reduce>(prim, inputs, quantType);
  } else if (op_type == "ReduceSum") {
    return NewPrimitiveC<Reduce>(prim, inputs, quantType);
  } else if (op_type == "ReduceSumSquare") {
    return NewPrimitiveC<Reduce>(prim, inputs, quantType);
  } else if (op_type == "Reshape") {
    return NewPrimitiveC<Reshape>(prim, inputs, quantType);
  } else if (op_type == "Slice") {
    return NewPrimitiveC<Slice>(prim, inputs, quantType);
  } else if (op_type == "Squeeze") {
    return NewPrimitiveC<Squeeze>(prim, inputs, quantType);
  } else if (op_type == "TensorAdd") {
    return NewPrimitiveC<Add>(prim, inputs, quantType);
  } else if (op_type == "Transpose") {
    return NewPrimitiveC<Transpose>(prim, inputs, quantType);
  } else if (op_type == "Elu") {
    return NewPrimitiveC<Elu>(prim, inputs, quantType);
  } else if (op_type == "Log") {
    return NewPrimitiveC<Log>(prim, inputs, quantType);
  } else if (op_type == "Exp") {
    return NewPrimitiveC<Exp>(prim, inputs, quantType);
  } else if (op_type == "Neg") {
    return NewPrimitiveC<Neg>(prim, inputs, quantType);
  } else if (op_type == "DeConv2D") {
    return NewPrimitiveC<DeConv2D>(prim, inputs, quantType);
  } else if (op_type == "tuple_getitem") {
    return NewPrimitiveC<TupleGetItem>(prim, inputs, quantType);
  } else if (op_type == "Softmax") {
    return NewPrimitiveC<SoftMax>(prim, inputs, quantType);
  } else if (op_type == "StridedSlice") {
    return NewPrimitiveC<StridedSlice>(prim, inputs, quantType);
  } else if (op_type == "Cast") {
    return NewPrimitiveC<Cast>(prim, inputs, quantType);
  } else if (op_type == "Maximum") {
    return NewPrimitiveC<Maximum>(prim, inputs, quantType);
  } else if (op_type == "Split") {
    return NewPrimitiveC<Split>(prim, inputs, quantType);
  } else if (op_type == "OneHot") {
    return NewPrimitiveC<OneHot>(prim, inputs, quantType);
  } else if (op_type == "Dropout") {
    return NewPrimitiveC<Dropout>(prim, inputs, quantType);
  } else if (op_type == "While") {
    return NewPrimitiveC<While>(prim, inputs, quantType);
  } else if (op_type == "MirrorPad") {
    return NewPrimitiveC<Pad>(prim, inputs, quantType);
  } else if (op_type == "GatherV2") {
    return NewPrimitiveC<Gather>(prim, inputs, quantType);
  } else if (op_type == "OnesLike") {
    return NewPrimitiveC<OnesLike>(prim, inputs, quantType);
  } else if (op_type == "Pow") {
    return NewPrimitiveC<Power>(prim, inputs, quantType);
  } else if (op_type == "Sub") {
    return NewPrimitiveC<Sub>(prim, inputs, quantType);
  } else if (op_type == "ExpandDims") {
    return NewPrimitiveC<ExpandDims>(prim, inputs, quantType);
  } else if (op_type == "UnsortedSegmentSum") {
    return NewPrimitiveC<UnsortedSegmentSum>(prim, inputs, quantType);
  } else if (op_type == "ResizeNearestNeighbor") {
    return NewPrimitiveC<Resize>(prim, inputs, quantType);
  } else if (op_type == "ResizeBilinear") {
    return NewPrimitiveC<Resize>(prim, inputs, quantType);
  } else if (op_type == "Floor") {
    return NewPrimitiveC<Floor>(prim, inputs, quantType);
  } else if (op_type == "Minimum") {
    return NewPrimitiveC<Minimum>(prim, inputs, quantType);
  } else if (op_type == "Div") {
    return NewPrimitiveC<Div>(prim, inputs, quantType);
  } else if (op_type == "Tanh") {
    return NewPrimitiveC<Activation>(prim, inputs, quantType);
  } else if (op_type == "Equal") {
    return NewPrimitiveC<Equal>(prim, inputs, quantType);
  } else if (op_type == "TopK") {
    return NewPrimitiveC<TopK>(prim, inputs, quantType);
  } else if (op_type == "Mod") {
    return NewPrimitiveC<Mod>(prim, inputs, quantType);
  } else if (op_type == "ArgMin" || op_type == "ArgMinWithValue") {
    return NewPrimitiveC<ArgMin>(prim, inputs, quantType);
  } else if (op_type == "Range") {
    return NewPrimitiveC<Range>(prim, inputs, quantType);
  } else if (op_type == "Tile") {
    return NewPrimitiveC<Tile>(prim, inputs, quantType);
  } else if (op_type == "GatherNd") {
    return NewPrimitiveC<GatherNd>(prim, inputs, quantType);
  } else if (op_type == "Square") {
    return NewPrimitiveC<Square>(prim, inputs, quantType);
  } else if (op_type == "Sqrt") {
    return NewPrimitiveC<Sqrt>(prim, inputs, quantType);
  } else if (op_type == "Greater") {
    return NewPrimitiveC<Greater>(prim, inputs, quantType);
  } else if (op_type == "Switch") {
    return NewPrimitiveC<Switch>(prim, inputs, quantType);
  } else if (op_type == "Partial") {
    return NewPrimitiveC<Partial>(prim, inputs, quantType);
  } else if (op_type == "Merge") {
    return NewPrimitiveC<Merge>(prim, inputs, quantType);
  } else if (op_type == "LayerNorm") {
    return NewPrimitiveC<LayerNorm>(prim, inputs, quantType);
  } else if (op_type == "ArgMax" || op_type == "ArgMaxWithValue") {
    return NewPrimitiveC<ArgMax>(prim, inputs, quantType);
  } else if (op_type == "Gelu") {
    return NewPrimitiveC<GeLU>(prim, inputs, quantType);

#ifdef SUPPORT_TRAIN
  } else if (op_type == "SoftmaxCrossEntropyWithLogits") {
    return NewPrimitiveC<SoftmaxCrossEntropy>(prim, inputs, quantType);
  } else if (op_type == "SparseSoftmaxCrossEntropyWithLogits") {
    return NewPrimitiveC<SparseSoftmaxCrossEntropy>(prim, inputs, quantType);
  } else if (op_type == "BiasAddGrad") {
    return NewPrimitiveC<BiasGrad>(prim, inputs, quantType);
  } else if (op_type == "ApplyMomentum") {
    return NewPrimitiveC<ApplyMomentum>(prim, inputs, quantType);
  } else if (op_type == "Depend") {
    return NewPrimitiveC<Depend>(prim, inputs, quantType);
  } else if (op_type == "ControlDepend") {
    return NewPrimitiveC<ControlDepend>(prim, inputs, quantType);
  } else if ((op_type == "ReluGrad" || op_type == "ReLU6Grad" || op_type == "SigmoidGrad" ||
              op_type == "HSigmoidGrad" || op_type == "HSwishGrad")) {
    return NewPrimitiveC<ActivationGrad>(prim, inputs, quantType);
  } else if ((op_type == "MaxPoolGrad") || (op_type == "AvgPoolGrad") || (op_type == "AvgPoolGradGpu") ||
             (op_type == "AvgPoolGradCpu")) {
    return NewPrimitiveC<PoolingGrad>(prim, inputs, quantType);
  } else if (op_type == "Conv2DBackpropFilter") {
    return NewPrimitiveC<Conv2DGradFilter>(prim, inputs, quantType);
  } else if (op_type == "Conv2DBackpropInput") {
    return NewPrimitiveC<Conv2DGradInput>(prim, inputs, quantType);
  } else if ((op_type == "BatchNormGrad") || (op_type == "FusedBatchNormGradEx")) {
    return NewPrimitiveC<BNGrad>(prim, inputs, quantType);
  } else if (op_type == "FlattenGrad") {
    return NewPrimitiveC<FlattenGrad>(prim, inputs, quantType);
  } else if ((op_type == "FusedBatchNormGrad") || (op_type == "FusedBatchNormGradCpu")) {
    return NewPrimitiveC<BNGrad>(prim, inputs, quantType);
  } else if (op_type == "PowerGrad") {
    return NewPrimitiveC<PowerGrad>(prim, inputs, quantType);
  } else if (op_type == "SGD") {
    return NewPrimitiveC<Sgd>(prim, inputs, quantType);
  } else if (op_type == "Adam") {
    return NewPrimitiveC<Adam>(prim, inputs, quantType);
  } else if (op_type == "Assign") {
    return NewPrimitiveC<Assign>(prim, inputs, quantType);
  } else if (op_type == "DropoutGrad") {
    return NewPrimitiveC<DropoutGrad>(prim, inputs, quantType);
  } else if (op_type == "MaximumGrad") {
    return NewPrimitiveC<MaximumGrad>(prim, inputs, quantType);
  } else if (op_type == "MinimumGrad") {
    return NewPrimitiveC<MinimumGrad>(prim, inputs, quantType);
  } else if (op_type == "AssignAdd") {
    return NewPrimitiveC<AssignAdd>(prim, inputs, quantType);
  } else if (op_type == "BinaryCrossEntropy") {
    return NewPrimitiveC<BinaryCrossEntropy>(prim, inputs, quantType);
  } else if (op_type == "BinaryCrossEntropyGrad") {
    return NewPrimitiveC<BinaryCrossEntropyGrad>(prim, inputs, quantType);
  } else if (op_type == "SmoothL1Loss") {
    return NewPrimitiveC<SmoothL1Loss>(prim, inputs, quantType);
  } else if (op_type == "SmoothL1LossGrad") {
    return NewPrimitiveC<SmoothL1LossGrad>(prim, inputs, quantType);
  } else if (op_type == "SigmoidCrossEntropyWithLogits") {
    return NewPrimitiveC<SigmoidCrossEntropyWithLogits>(prim, inputs, quantType);
  } else if (op_type == "SigmoidCrossEntropyWithLogitsGrad") {
    return NewPrimitiveC<SigmoidCrossEntropyWithLogitsGrad>(prim, inputs, quantType);
  } else if (op_type == "Pad") {
    return NewPrimitiveC<Pad>(prim, inputs, quantType);
#else
  } else if (op_type == "Conv2DBackpropInput") {
    return NewPrimitiveC<DeConv2D>(prim, inputs, quantType);
#endif
  } else {
    MS_LOG(ERROR) << "Unsupported primitive type in Create : " << op_type;
    return nullptr;
  }
}

PrimitiveC *PrimitiveC::Create(mindspore::schema::PrimitiveT *primitive) {
  MS_ASSERT(primitive != nullptr);
  auto op_type = primitive->value.type;
  switch (op_type) {
    case schema::PrimitiveType_SoftMax:
      return new (std::nothrow) SoftMax(primitive);
    case schema::PrimitiveType_Activation:
      return new (std::nothrow) Activation(primitive);
    case schema::PrimitiveType_Conv2D:
      return new (std::nothrow) Conv2D(primitive);
    case schema::PrimitiveType_DeConv2D:
      return new (std::nothrow) DeConv2D(primitive);
    case schema::PrimitiveType_Reduce:
      return new (std::nothrow) Reduce(primitive);
    case schema::PrimitiveType_Pooling:
      return new (std::nothrow) Pooling(primitive);
    case schema::PrimitiveType_ROIPooling:
      return new (std::nothrow) ROIPooling(primitive);
    case schema::PrimitiveType_DepthwiseConv2D:
      return new (std::nothrow) DepthwiseConv2D(primitive);
    case schema::PrimitiveType_FusedBatchNorm:
      return new (std::nothrow) FusedBatchNorm(primitive);
    case schema::PrimitiveType_BatchNorm:
      return new (std::nothrow) BatchNorm(primitive);
    case schema::PrimitiveType_FullConnection:
      return new (std::nothrow) FullConnection(primitive);
    case schema::PrimitiveType_Power:
      return new (std::nothrow) Power(primitive);
    case schema::PrimitiveType_Pad:
      return new (std::nothrow) Pad(primitive);
    case schema::PrimitiveType_Range:
      return new (std::nothrow) Range(primitive);
    case schema::PrimitiveType_Mul:
      return new (std::nothrow) Mul(primitive);
    case schema::PrimitiveType_Add:
      return new (std::nothrow) Add(primitive);
    case schema::PrimitiveType_Sub:
      return new (std::nothrow) Sub(primitive);
    case schema::PrimitiveType_Div:
      return new (std::nothrow) Div(primitive);
    case schema::PrimitiveType_BiasAdd:
      return new (std::nothrow) BiasAdd(primitive);
    case schema::PrimitiveType_ExpandDims:
      return new (std::nothrow) ExpandDims(primitive);
    case schema::PrimitiveType_ArgMax:
      return new (std::nothrow) ArgMax(primitive);
    case schema::PrimitiveType_ArgMin:
      return new (std::nothrow) ArgMin(primitive);
    case schema::PrimitiveType_Cast:
      return new (std::nothrow) Cast(primitive);
    case schema::PrimitiveType_Reshape:
      return new (std::nothrow) Reshape(primitive);
    case schema::PrimitiveType_Scale:
      return new (std::nothrow) Scale(primitive);
    case schema::PrimitiveType_Eltwise:
      return new (std::nothrow) Eltwise(primitive);
    case schema::PrimitiveType_Ceil:
      return new (std::nothrow) Ceil(primitive);
    case schema::PrimitiveType_Concat:
      return new (std::nothrow) Concat(primitive);
    case schema::PrimitiveType_Fill:
      return new (std::nothrow) Fill(primitive);
    case schema::PrimitiveType_Nhwc2Nchw:
      return new (std::nothrow) Nhwc2Nchw(primitive);
    case schema::PrimitiveType_Nchw2Nhwc:
      return new (std::nothrow) Nchw2Nhwc(primitive);
    case schema::PrimitiveType_Transpose:
      return new (std::nothrow) Transpose(primitive);
    case schema::PrimitiveType_Slice:
      return new (std::nothrow) Slice(primitive);
    case schema::PrimitiveType_Squeeze:
      return new (std::nothrow) Squeeze(primitive);
    case schema::PrimitiveType_Flatten:
      return new (std::nothrow) Flatten(primitive);
    case schema::PrimitiveType_Stack:
      return new (std::nothrow) Stack(primitive);
    case schema::PrimitiveType_Crop:
      return new (std::nothrow) Crop(primitive);
    case schema::PrimitiveType_SquaredDifference:
      return new (std::nothrow) SquaredDifference(primitive);
    case schema::PrimitiveType_AddN:
      return new (std::nothrow) AddN(primitive);
    case schema::PrimitiveType_Abs:
      return new (std::nothrow) Abs(primitive);
    case schema::PrimitiveType_Sin:
      return new (std::nothrow) Sin(primitive);
    case schema::PrimitiveType_Cos:
      return new (std::nothrow) Cos(primitive);
    case schema::PrimitiveType_Log:
      return new (std::nothrow) Log(primitive);
    case schema::PrimitiveType_Sqrt:
      return new (std::nothrow) Sqrt(primitive);
    case schema::PrimitiveType_Rsqrt:
      return new (std::nothrow) Rsqrt(primitive);
    case schema::PrimitiveType_Square:
      return new (std::nothrow) Square(primitive);
    case schema::PrimitiveType_Exp:
      return new (std::nothrow) Exp(primitive);
    case schema::PrimitiveType_Gather:
      return new (std::nothrow) Gather(primitive);
    case schema::PrimitiveType_GatherNd:
      return new (std::nothrow) GatherNd(primitive);
    case schema::PrimitiveType_LocalResponseNormalization:
      return new (std::nothrow) LocalResponseNormalization(primitive);
    case schema::PrimitiveType_Maximum:
      return new (std::nothrow) Maximum(primitive);
    case schema::PrimitiveType_Minimum:
      return new (std::nothrow) Minimum(primitive);
    case schema::PrimitiveType_StridedSlice:
      return new (std::nothrow) StridedSlice(primitive);
    case schema::PrimitiveType_LeakyReLU:
      return new (std::nothrow) LeakyReLU(primitive);
    case schema::PrimitiveType_PReLU:
      return new (std::nothrow) PReLU(primitive);
    case schema::PrimitiveType_Round:
      return new (std::nothrow) Round(primitive);
    case schema::PrimitiveType_Reverse:
      return new (std::nothrow) Reverse(primitive);
    case schema::PrimitiveType_ReverseSequence:
      return new (std::nothrow) ReverseSequence(primitive);
    case schema::PrimitiveType_LogicalAnd:
      return new (std::nothrow) LogicalAnd(primitive);
    case schema::PrimitiveType_LogicalOr:
      return new (std::nothrow) LogicalOr(primitive);
    case schema::PrimitiveType_LogicalNot:
      return new (std::nothrow) LogicalNot(primitive);
    case schema::PrimitiveType_FloorDiv:
      return new (std::nothrow) FloorDiv(primitive);
    case schema::PrimitiveType_FloorMod:
      return new (std::nothrow) FloorMod(primitive);
    case schema::PrimitiveType_Mod:
      return new (std::nothrow) Mod(primitive);
    case schema::PrimitiveType_Equal:
      return new (std::nothrow) Equal(primitive);
    case schema::PrimitiveType_NotEqual:
      return new (std::nothrow) NotEqual(primitive);
    case schema::PrimitiveType_Less:
      return new (std::nothrow) Less(primitive);
    case schema::PrimitiveType_LessEqual:
      return new (std::nothrow) LessEqual(primitive);
    case schema::PrimitiveType_Greater:
      return new (std::nothrow) Greater(primitive);
    case schema::PrimitiveType_GreaterEqual:
      return new (std::nothrow) GreaterEqual(primitive);
    case schema::PrimitiveType_Floor:
      return new (std::nothrow) Floor(primitive);
    case schema::PrimitiveType_Split:
      return new (std::nothrow) Split(primitive);
    case schema::PrimitiveType_OneHot:
      return new (std::nothrow) OneHot(primitive);
    case schema::PrimitiveType_PriorBox:
      return new (std::nothrow) PriorBox(primitive);
    case schema::PrimitiveType_SpaceToDepth:
      return new (std::nothrow) SpaceToDepth(primitive);
    case schema::PrimitiveType_Tile:
      return new (std::nothrow) Tile(primitive);
    case schema::PrimitiveType_Resize:
      return new (std::nothrow) Resize(primitive);
    case schema::PrimitiveType_Unstack:
      return new (std::nothrow) Unstack(primitive);
    case schema::PrimitiveType_Unique:
      return new (std::nothrow) Unique(primitive);
    case schema::PrimitiveType_TopK:
      return new (std::nothrow) TopK(primitive);
    case schema::PrimitiveType_MatMul:
      return new (std::nothrow) MatMul(primitive);
    case schema::PrimitiveType_QuantDTypeCast:
      return new (std::nothrow) QuantDTypeCast(primitive);
    case schema::PrimitiveType_EmbeddingLookup:
      return new (std::nothrow) EmbeddingLookup(primitive);
    case schema::PrimitiveType_Elu:
      return new (std::nothrow) Elu(primitive);
    case schema::PrimitiveType_DeDepthwiseConv2D:
      return new (std::nothrow) DeDepthwiseConv2D(primitive);
    case schema::PrimitiveType_Shape:
      return new (std::nothrow) Shape(primitive);
    case schema::PrimitiveType_Unsqueeze:
      return new (std::nothrow) Unsqueeze(primitive);
    case schema::PrimitiveType_BatchToSpace:
    case schema::PrimitiveType_BatchToSpaceND:
      return new (std::nothrow) BatchToSpace(primitive);
    case schema::PrimitiveType_SpaceToBatch:
      return new (std::nothrow) SpaceToBatch(primitive);
    case schema::PrimitiveType_SpaceToBatchND:
      return new (std::nothrow) SpaceToBatchND(primitive);
    case schema::PrimitiveType_BroadcastTo:
      return new (std::nothrow) BroadcastTo(primitive);
    case schema::PrimitiveType_DepthToSpace:
      return new (std::nothrow) DepthToSpace(primitive);
    case schema::PrimitiveType_Lstm:
      return new (std::nothrow) Lstm(primitive);
    case schema::PrimitiveType_ZerosLike:
      return new (std::nothrow) ZerosLike(primitive);
    case schema::PrimitiveType_MakeTuple:
      return new (std::nothrow) MakeTuple(primitive);
    case schema::PrimitiveType_Where:
      return new (std::nothrow) Where(primitive);
    case schema::PrimitiveType_ScatterND:
      return new (std::nothrow) ScatterND(primitive);
    case schema::PrimitiveType_ConstantOfShape:
      return new (std::nothrow) ConstantOfShape(primitive);
    case schema::PrimitiveType_L2Norm:
      return new (std::nothrow) L2Norm(primitive);
    case schema::PrimitiveType_SparseToDense:
      return new (std::nothrow) SparseToDense(primitive);
    case schema::PrimitiveType_DetectionPostProcess:
      return new (std::nothrow) DetectionPostProcess(primitive);
    case schema::PrimitiveType_Dropout:
      return new (std::nothrow) Dropout(primitive);
    case schema::PrimitiveType_Neg:
      return new (std::nothrow) Neg(primitive);
    case schema::PrimitiveType_RealDiv:
      return new (std::nothrow) RealDiv(primitive);
    case schema::PrimitiveType_LshProjection:
      return new (std::nothrow) LshProjection(primitive);
    case schema::PrimitiveType_HashtableLookup:
      return new (std::nothrow) HashtableLookup(primitive);
    case schema::PrimitiveType_SkipGram:
      return new (std::nothrow) SkipGram(primitive);
    case schema::PrimitiveType_Clip:
      return new (std::nothrow) Clip(primitive);
    case schema::PrimitiveType_Adder:
      return new (std::nothrow) Adder(primitive);
    case schema::PrimitiveType_CustomPredict:
      return new (std::nothrow) CustomPredict(primitive);
    case schema::PrimitiveType_CustomNormalize:
      return new (std::nothrow) CustomNormalize(primitive);
    case schema::PrimitiveType_CustomExtractFeatures:
      return new (std::nothrow) CustomExtractFeatures(primitive);
    case schema::PrimitiveType_Upsample:
      return new (std::nothrow) Upsample(primitive);
    case schema::PrimitiveType_LayerNorm:
      return new (std::nothrow) LayerNorm(primitive);
    case schema::PrimitiveType_NonMaxSuppression:
      return new (std::nothrow) NonMaxSuppression(primitive);
    case schema::PrimitiveType_Identity:
      return new (std::nothrow) Identity(primitive);
    case schema::PrimitiveType_Rfft:
      return new (std::nothrow) Rfft(primitive);
    case schema::PrimitiveType_FftReal:
      return new (std::nothrow) FftReal(primitive);
    case schema::PrimitiveType_FftImag:
      return new (std::nothrow) FftImag(primitive);
    case schema::PrimitiveType_AudioSpectrogram:
      return new (std::nothrow) AudioSpectrogram(primitive);
    case schema::PrimitiveType_Mfcc:
      return new (std::nothrow) Mfcc(primitive);
    case schema::PrimitiveType_InstanceNorm:
      return new (std::nothrow) InstanceNorm(primitive);
    case schema::PrimitiveType_While:
      return new (std::nothrow) While(primitive);
    case schema::PrimitiveType_OnnxInt8Quantize:
      return new (std::nothrow) Quant(primitive);
    case schema::PrimitiveType_OnnxInt8Dequantize:
      return new (std::nothrow) Dequant(primitive);
    case schema::PrimitiveType_Reciprocal:
      return new (std::nothrow) Reciprocal(primitive);
    case schema::PrimitiveType_Constant:
      return new (std::nothrow) Constant(primitive);
    case schema::PrimitiveType_TensorListFromTensor:
      return new (std::nothrow) TensorListFromTensor(primitive);
    case schema::PrimitiveType_TensorListGetItem:
      return new (std::nothrow) TensorListGetItem(primitive);
    case schema::PrimitiveType_TensorListSetItem:
      return new (std::nothrow) TensorListSetItem(primitive);
    case schema::PrimitiveType_TensorListReserve:
      return new (std::nothrow) TensorListReserve(primitive);
    case schema::PrimitiveType_TensorListStack:
      return new (std::nothrow) TensorListStack(primitive);
    case schema::PrimitiveType_Switch:
      return new (std::nothrow) Switch(primitive);
    case schema::PrimitiveType_Merge:
      return new (std::nothrow) Merge(primitive);
    case schema::PrimitiveType_Partial:
      return new (std::nothrow) Partial(primitive);
    case schema::PrimitiveType_Assert:
      return new (std::nothrow) AssertOP(primitive);
    case schema::PrimitiveType_GeLU:
      return new (std::nothrow) GeLU(primitive);
    case schema::PrimitiveType_If:
      return new (std::nothrow) If(primitive);
    case schema::PrimitiveType_Select:
      return new (std::nothrow) Select(primitive);
    case schema::PrimitiveType_Gru:
      return new (std::nothrow) Gru(primitive);
    case schema::PrimitiveType_Size:
      return new (std::nothrow) Size(primitive);
    case schema::PrimitiveType_InvertPermutation:
      return new (std::nothrow) InvertPermutation(primitive);
    case schema::PrimitiveType_RandomStandardNormal:
      return new (std::nothrow) RandomStandardNormal(primitive);
    case schema::PrimitiveType_CropAndResize:
      return new (std::nothrow) CropAndResize(primitive);
    case schema::PrimitiveType_NonZero:
      return new (std::nothrow) NonZero(primitive);
#ifdef SUPPORT_TRAIN
    case schema::PrimitiveType_ActivationGrad:
      return new (std::nothrow) ActivationGrad(primitive);
    case schema::PrimitiveType_PoolingGrad:
      return new (std::nothrow) PoolingGrad(primitive);
    case schema::PrimitiveType_Conv2DGradFilter:
      return new (std::nothrow) Conv2DGradFilter(primitive);
    case schema::PrimitiveType_Conv2DGradInput:
      return new (std::nothrow) Conv2DGradInput(primitive);
    case schema::PrimitiveType_GroupConv2DGradInput:
      return new (std::nothrow) GroupConv2DGradInput(primitive);
    case schema::PrimitiveType_BiasGrad:
      return new (std::nothrow) BiasGrad(primitive);
    case schema::PrimitiveType_ApplyMomentum:
      return new (std::nothrow) ApplyMomentum(primitive);
    case schema::PrimitiveType_BNGrad:
      return new (std::nothrow) BNGrad(primitive);
    case schema::PrimitiveType_AddGrad:
      return new (std::nothrow) ArithmeticGrad(primitive);
    case schema::PrimitiveType_SubGrad:
      return new (std::nothrow) ArithmeticGrad(primitive);
    case schema::PrimitiveType_MulGrad:
      return new (std::nothrow) ArithmeticGrad(primitive);
    case schema::PrimitiveType_DivGrad:
      return new (std::nothrow) ArithmeticGrad(primitive);
    case schema::PrimitiveType_SoftmaxCrossEntropy:
      return new (std::nothrow) SoftmaxCrossEntropy(primitive);
    case schema::PrimitiveType_SparseSoftmaxCrossEntropy:
      return new (std::nothrow) SparseSoftmaxCrossEntropy(primitive);
    case schema::PrimitiveType_PowerGrad:
      return new (std::nothrow) PowerGrad(primitive);
    case schema::PrimitiveType_Depend:
      return new (std::nothrow) Depend(primitive);
    case schema::PrimitiveType_ControlDepend:
      return new (std::nothrow) ControlDepend(primitive);
    case schema::PrimitiveType_FlattenGrad:
      return new (std::nothrow) FlattenGrad(primitive);
    case schema::PrimitiveType_NegGrad:
      return new (std::nothrow) NegGrad(primitive);
    case schema::PrimitiveType_LogGrad:
      return new (std::nothrow) LogGrad(primitive);
    case schema::PrimitiveType_Sgd:
      return new (std::nothrow) Sgd(primitive);
    case schema::PrimitiveType_Adam:
      return new (std::nothrow) Adam(primitive);
    case schema::PrimitiveType_Assign:
      return new (std::nothrow) Assign(primitive);
    case schema::PrimitiveType_AssignAdd:
      return new (std::nothrow) AssignAdd(primitive);
    case schema::PrimitiveType_OnesLike:
      return new (std::nothrow) OnesLike(primitive);
    case schema::PrimitiveType_UnsortedSegmentSum:
      return new (std::nothrow) UnsortedSegmentSum(primitive);
    case schema::PrimitiveType_BinaryCrossEntropyGrad:
      return new (std::nothrow) BinaryCrossEntropyGrad(primitive);
    case schema::PrimitiveType_BinaryCrossEntropy:
      return new (std::nothrow) BinaryCrossEntropy(primitive);
    case schema::PrimitiveType_DropoutGrad:
      return new (std::nothrow) DropoutGrad(primitive);
    case schema::PrimitiveType_MaximumGrad:
      return new (std::nothrow) MaximumGrad(primitive);
    case schema::PrimitiveType_MinimumGrad:
      return new (std::nothrow) MinimumGrad(primitive);
    case schema::PrimitiveType_SmoothL1Loss:
      return new (std::nothrow) SmoothL1Loss(primitive);
    case schema::PrimitiveType_SmoothL1LossGrad:
      return new (std::nothrow) SmoothL1LossGrad(primitive);
    case schema::PrimitiveType_SigmoidCrossEntropyWithLogits:
      return new (std::nothrow) SigmoidCrossEntropyWithLogits(primitive);
    case schema::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad:
      return new (std::nothrow) SigmoidCrossEntropyWithLogitsGrad(primitive);
#endif
    default:
      MS_LOG(ERROR) << "Unsupported primitive type in Create : " << schema::EnumNamePrimitiveType(op_type);
      break;
  }
  return nullptr;
}

#else
void PrimitiveC::set_quant_type(schema::QuantType quant_type) { this->quant_type_ = quant_type; }
schema::QuantType PrimitiveC::quant_type() const { return quant_type_; }
#endif

int PrimitiveC::Type() const {
  if (this->primitive_ == nullptr && this->op_type_ == OP_TYPE_NOT_SET) {
    return schema::PrimitiveType_NONE;
  }
#ifdef PRIMITIVE_WRITEABLE
  if (op_type_ != OP_TYPE_NOT_SET) {
    return op_type_;
  }
  return this->primitive_->value.type;
#else
  return this->primitive_->value_type();
#endif
}
bool PrimitiveC::infer_flag() const { return this->infer_flag_; }

void PrimitiveC::set_infer_flag(bool flag) { this->infer_flag_ = flag; }

int PrimitiveC::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  auto input = inputs.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs.front();
  MS_ASSERT(output != nullptr);
  output->set_shape(input->shape());
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  return 0;
}

}  // namespace lite
}  // namespace mindspore
