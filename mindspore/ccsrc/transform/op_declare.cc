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

#include "transform/op_declare.h"

#include <vector>

#include "transform/all_ops.h"
#include "utils/utils.h"

namespace mindspore {
namespace transform {
#define INPUT_MAP(T) \
  template <>        \
  const std::unordered_map<int, InputDesc> OpAdapter<T>::input_map_
#define EMPTY_INPUT_MAP std::unordered_map<int, InputDesc>()
#define INPUT_DESC(name) \
  {                      \
#name, \
    [](const OperatorPtr op, const OperatorPtr input) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_input_##name(*input); \
    }, \
    [](const OperatorPtr op, const OutHandler& handle) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_input_##name(*(handle.op), handle.out); \
    }, \
    [](const OperatorPtr op, const GeTensorDesc desc) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->update_input_desc_##name(desc); \
    }                 \
  }

#define DYN_INPUT_MAP(T) \
  template <>            \
  const std::unordered_map<int, DynInputDesc> OpAdapter<T>::dyn_input_map_
#define DYN_INPUT_DESC(name) \
  {                          \
#name, \
    [](const OperatorPtr op, unsigned int num) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->create_dynamic_input_##name(num); \
    }, \
    [](const OperatorPtr op, unsigned int index, const OperatorPtr input) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_dynamic_input_##name(index, *input); \
    }, \
    [](const OperatorPtr op, unsigned int index, const OutHandler& handle) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_dynamic_input_##name(index, *(handle.op), handle.out); \
    }                     \
  }

#define ATTR_MAP(T) \
  template <>       \
  const std::unordered_map<std::string, AttrDesc> OpAdapter<T>::attr_map_
#define EMPTY_ATTR_MAP std::unordered_map<std::string, AttrDesc>()
#define ATTR_DESC(name, ...) \
  {                          \
#name, \
    [](const OperatorPtr op, const ValuePtr& value) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_attr_##name(ConvertAny(value, __VA_ARGS__)); \
    }                     \
  }

#define INPUT_ATTR_MAP(T) \
  template <>             \
  const std::unordered_map<unsigned int, AttrDesc> OpAdapter<T>::input_attr_map_

#define OUTPUT_MAP(T) \
  template <>         \
  const std::unordered_map<int, OutputDesc> OpAdapter<T>::output_map_
#define OUTPUT_DESC(name) \
  {                       \
#name, \
    [](const OperatorPtr op, const GeTensorDesc desc) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->update_output_desc_##name(desc); \
    }                  \
  }

#define DYN_OUTPUT_MAP(T) \
  template <>             \
  const std::unordered_map<int, DynOutputDesc> OpAdapter<T>::dyn_output_map_

#define DYN_OUTPUT_DESC(name) \
  {                           \
#name, \
    [](const OperatorPtr op, unsigned int num) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->create_dynamic_output_##name(num); \
    }                      \
  }

template <>
std::unordered_map<std::string, std::unordered_map<int, std::string>> OpAdapter<ge::Operator>::cus_input_map_{};
template <>
std::unordered_map<std::string, std::unordered_map<int, std::string>> OpAdapter<ge::Operator>::cus_output_map_{};

// --------------specialization for each operator----------
// const
INPUT_MAP(Const) = EMPTY_INPUT_MAP;
ATTR_MAP(Const) = {{"value", ATTR_DESC(value, AnyTraits<AnyValue>())}};
OUTPUT_MAP(Const) = {{0, OUTPUT_DESC(y)}};

// Assign
INPUT_MAP(Assign) = {{1, INPUT_DESC(ref)}, {2, INPUT_DESC(value)}};
ATTR_MAP(Assign) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Assign) = {{0, OUTPUT_DESC(ref)}};

// Constant
INPUT_MAP(Constant) = EMPTY_INPUT_MAP;
ATTR_MAP(Constant) = {{"value", ATTR_DESC(value, AnyTraits<AnyValue>())}};
OUTPUT_MAP(Constant) = {{0, OUTPUT_DESC(y)}};

// ApplyMomentum
INPUT_MAP(ApplyMomentum) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(lr)}, {4, INPUT_DESC(grad)}, {5, INPUT_DESC(momentum)}};
ATTR_MAP(ApplyMomentum) = {{"use_nesterov", ATTR_DESC(use_nesterov, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyMomentum) = {{0, OUTPUT_DESC(var)}};

// ScalarSummary
INPUT_MAP(Summary) = {{2, INPUT_DESC(x)}};
ATTR_MAP(Summary) = EMPTY_ATTR_MAP;

// Data
INPUT_MAP(Data) = EMPTY_INPUT_MAP;
ATTR_MAP(Data) = EMPTY_ATTR_MAP;

// BatchNorm
INPUT_MAP(BatchNorm) = {{1, INPUT_DESC(x)},
                        {2, INPUT_DESC(scale)},
                        {3, INPUT_DESC(offset)},
                        {4, INPUT_DESC(mean)},
                        {5, INPUT_DESC(variance)}};
ATTR_MAP(BatchNorm) = {{"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                       {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                       {"is_training", ATTR_DESC(is_training, AnyTraits<bool>())}};
OUTPUT_MAP(BatchNorm) = {{0, OUTPUT_DESC(y)},
                         {1, OUTPUT_DESC(batch_mean)},
                         {2, OUTPUT_DESC(batch_variance)},
                         {3, OUTPUT_DESC(reserve_space_1)},
                         {4, OUTPUT_DESC(reserve_space_2)}};

// BatchNormGrad
INPUT_MAP(BatchNormGrad) = {{1, INPUT_DESC(y_backprop)},
                            {2, INPUT_DESC(x)},
                            {3, INPUT_DESC(scale)},
                            {4, INPUT_DESC(reserve_space_1)},
                            {5, INPUT_DESC(reserve_space_2)}};
ATTR_MAP(BatchNormGrad) = {{"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                           {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                           {"is_training", ATTR_DESC(is_training, AnyTraits<bool>())}};
OUTPUT_MAP(BatchNormGrad) = {{0, OUTPUT_DESC(x_backprop)},
                             {1, OUTPUT_DESC(scale_backprop)},
                             {2, OUTPUT_DESC(offset_backprop)},
                             {3, OUTPUT_DESC(reserve_space_4)},
                             {4, OUTPUT_DESC(reserve_space_5)}};

// Relu
INPUT_MAP(Relu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Relu) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Relu) = {{0, OUTPUT_DESC(y)}};

// Elu
INPUT_MAP(Elu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Elu) = {{"alpha", ATTR_DESC(alpha, AnyTraits<float>())}};
OUTPUT_MAP(Elu) = {{0, OUTPUT_DESC(y)}};

// EluGrad
INPUT_MAP(EluGrad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(activations)}};
ATTR_MAP(EluGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(EluGrad) = {{0, OUTPUT_DESC(y)}};

// PRelu
INPUT_MAP(PRelu) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(weight)}};
ATTR_MAP(PRelu) = EMPTY_ATTR_MAP;
OUTPUT_MAP(PRelu) = {{0, OUTPUT_DESC(y)}};

// PReluGrad
INPUT_MAP(PReluGrad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(features)}, {3, INPUT_DESC(weights)}};
ATTR_MAP(PReluGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(PReluGrad) = {{0, OUTPUT_DESC(dx)}, {1, OUTPUT_DESC(da)}};

// Sigmoid
INPUT_MAP(Sigmoid) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Sigmoid) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sigmoid) = {{0, OUTPUT_DESC(y)}};

// SigmoidGrad
INPUT_MAP(SigmoidGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(SigmoidGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SigmoidGrad) = {{0, OUTPUT_DESC(z)}};

// L2NormalizeGrad
INPUT_MAP(L2NormalizeGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(dy)}};
ATTR_MAP(L2NormalizeGrad) = {
  {"axis", ATTR_DESC(dim, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"epsilon", ATTR_DESC(eps, AnyTraits<float>())}};
OUTPUT_MAP(L2NormalizeGrad) = {{0, OUTPUT_DESC(dx)}};

// LarsV2Update
INPUT_MAP(LarsV2Update) = {{1, INPUT_DESC(w)},
                           {2, INPUT_DESC(g)},
                           {3, INPUT_DESC(w_square_sum)},
                           {4, INPUT_DESC(g_square_sum)},
                           {5, INPUT_DESC(weight_decay)},
                           {6, INPUT_DESC(learning_rate)}};
ATTR_MAP(LarsV2Update) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                          {"hyperpara", ATTR_DESC(hyperpara, AnyTraits<float>())},
                          {"use_clip", ATTR_DESC(use_clip, AnyTraits<bool>())}};
OUTPUT_MAP(LarsV2Update) = {{0, OUTPUT_DESC(g_new)}};

// L2Normalize
INPUT_MAP(L2Normalize) = {{1, INPUT_DESC(x)}};
ATTR_MAP(L2Normalize) = {
  {"axis", ATTR_DESC(axis, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"epsilon", ATTR_DESC(eps, AnyTraits<float>())}};
OUTPUT_MAP(L2Normalize) = {{0, OUTPUT_DESC(y)}};

// CumsumD
INPUT_MAP(CumsumD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(CumsumD) = {{2, ATTR_DESC(axis, AnyTraits<int64_t>())}};
ATTR_MAP(CumsumD) = {{"exclusive", ATTR_DESC(exclusive, AnyTraits<bool>())},
                     {"reverse", ATTR_DESC(reverse, AnyTraits<bool>())}};
OUTPUT_MAP(CumsumD) = {{0, OUTPUT_DESC(y)}};

// SoftmaxV2
INPUT_MAP(SoftmaxV2) = {{1, INPUT_DESC(x)}};
ATTR_MAP(SoftmaxV2) = {
  {"axis", ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
};
OUTPUT_MAP(SoftmaxV2) = {{0, OUTPUT_DESC(y)}};

// SoftmaxGrad
INPUT_MAP(SoftmaxGrad) = {{1, INPUT_DESC(softmax)}, {2, INPUT_DESC(grad_softmax)}};
OUTPUT_MAP(SoftmaxGrad) = {{0, OUTPUT_DESC(grad_x)}};
ATTR_MAP(SoftmaxGrad) = EMPTY_ATTR_MAP;

// Flatten
INPUT_MAP(Flatten) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Flatten) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Flatten) = {{0, OUTPUT_DESC(y)}};

// add
INPUT_MAP(Add) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Add) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Add) = {{0, OUTPUT_DESC(y)}};

// GatherV2
INPUT_MAP(GatherV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(axis)}};
ATTR_MAP(GatherV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherV2) = {{0, OUTPUT_DESC(y)}};

// ReduceSumD
INPUT_MAP(ReduceSumD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceSumD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceSumD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceSumD) = {{0, OUTPUT_DESC(y)}};

// ReduceProdD
INPUT_MAP(ReduceProdD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceProdD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceProdD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceProdD) = {{0, OUTPUT_DESC(y)}};

// CumprodD
INPUT_MAP(CumprodD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(CumprodD) = {{2, ATTR_DESC(axis, AnyTraits<int64_t>())}};
ATTR_MAP(CumprodD) = {{"exclusive", ATTR_DESC(exclusive, AnyTraits<bool>())},
                      {"reverse", ATTR_DESC(reverse, AnyTraits<bool>())}};
OUTPUT_MAP(CumprodD) = {{0, OUTPUT_DESC(y)}};

// SoftmaxCrossEntropyWithLogits
INPUT_MAP(SoftmaxCrossEntropyWithLogits) = {{1, INPUT_DESC(features)}, {2, INPUT_DESC(labels)}};
ATTR_MAP(SoftmaxCrossEntropyWithLogits) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SoftmaxCrossEntropyWithLogits) = {{0, OUTPUT_DESC(loss)}, {1, OUTPUT_DESC(backprop)}};

// MeanGrad
INPUT_MAP(MeanGrad) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(MeanGrad) = {{2, ATTR_DESC(mean_grad_output_shape_value, kOpFormat_NHWC,
                                          AnyTraits<std::vector<int64_t>>(), AnyTraits<int64_t>())}};
ATTR_MAP(MeanGrad) = {{"mode", ATTR_DESC(mode, AnyTraits<int64_t>())}};

INPUT_MAP(SliceD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(SliceD) = {{2, ATTR_DESC(offsets, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                          {3, ATTR_DESC(size, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(SliceD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SliceD) = {{0, OUTPUT_DESC(y)}};

// MaxPool
INPUT_MAP(MaxPool) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MaxPool) = {{"ksize", ATTR_DESC(ksize, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                     {"strides", ATTR_DESC(strides, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                     {"padding", ATTR_DESC(padding, AnyTraits<std::string>())},
                     {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPool) = {{0, OUTPUT_DESC(y)}};

// AvgPool
INPUT_MAP(AvgPool) = {{1, INPUT_DESC(x)}};
ATTR_MAP(AvgPool) = {{"ksize", ATTR_DESC(ksize, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                     {"strides", ATTR_DESC(strides, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                     {"padding", ATTR_DESC(padding, AnyTraits<std::string>())},
                     {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(AvgPool) = {{0, OUTPUT_DESC(y)}};

// GreaterEqual
INPUT_MAP(GreaterEqual) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(GreaterEqual) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GreaterEqual) = {{0, OUTPUT_DESC(y)}};

// AssignAdd
INPUT_MAP(AssignAdd) = {{1, INPUT_DESC(ref)}, {2, INPUT_DESC(value)}};
ATTR_MAP(AssignAdd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AssignAdd) = {{0, OUTPUT_DESC(ref)}};

// AssignSub
INPUT_MAP(AssignSub) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(value)}};
ATTR_MAP(AssignSub) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AssignSub) = {{0, OUTPUT_DESC(var)}};

// Cos
INPUT_MAP(Cos) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Cos) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Cos) = {{0, OUTPUT_DESC(y)}};

// Acos
INPUT_MAP(Acos) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Acos) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Acos) = {{0, OUTPUT_DESC(y)}};

// AcosGrad
INPUT_MAP(AcosGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(AcosGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AcosGrad) = {{0, OUTPUT_DESC(z)}};

// Acosh
INPUT_MAP(Acosh) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Acosh) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Acosh) = {{0, OUTPUT_DESC(y)}};

// AcoshGrad
INPUT_MAP(AcoshGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(AcoshGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AcoshGrad) = {{0, OUTPUT_DESC(z)}};

// Floor
INPUT_MAP(Floor) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Floor) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Floor) = {{0, OUTPUT_DESC(y)}};

// FloorDiv
INPUT_MAP(FloorDiv) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(FloorDiv) = EMPTY_ATTR_MAP;
OUTPUT_MAP(FloorDiv) = {{0, OUTPUT_DESC(y)}};

// FloorMod
INPUT_MAP(FloorMod) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(FloorMod) = EMPTY_ATTR_MAP;
OUTPUT_MAP(FloorMod) = {{0, OUTPUT_DESC(y)}};

// Sin
INPUT_MAP(Sin) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Sin) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sin) = {{0, OUTPUT_DESC(y)}};

// Exp
INPUT_MAP(Exp) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Exp) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Exp) = {{0, OUTPUT_DESC(y)}};

// BoundingBoxEncode
INPUT_MAP(BoundingBoxEncode) = {
  {1, INPUT_DESC(anchor_box)},
  {2, INPUT_DESC(ground_truth_box)},
};
ATTR_MAP(BoundingBoxEncode) = {
  {"means", ATTR_DESC(means, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
  {"stds", ATTR_DESC(stds, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
};
OUTPUT_MAP(BoundingBoxEncode) = {{0, OUTPUT_DESC(delats)}};

// BoundingBoxDecode
INPUT_MAP(BoundingBoxDecode) = {
  {1, INPUT_DESC(rois)},
  {2, INPUT_DESC(deltas)},
};
ATTR_MAP(BoundingBoxDecode) = {
  {"means", ATTR_DESC(means, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
  {"stds", ATTR_DESC(stds, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
  {"max_shape", ATTR_DESC(max_shape, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"wh_ratio_clip", ATTR_DESC(wh_ratio_clip, AnyTraits<float>())},
};
OUTPUT_MAP(BoundingBoxDecode) = {{0, OUTPUT_DESC(bboxes)}};

// TopK
INPUT_MAP(TopK) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(k)}};
ATTR_MAP(TopK) = {{"sorted", ATTR_DESC(sorted, AnyTraits<bool>())}};
OUTPUT_MAP(TopK) = {{0, OUTPUT_DESC(values)}, {1, OUTPUT_DESC(indices)}};

// Multiply
INPUT_MAP(Multiply) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}};
ATTR_MAP(Multiply) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Multiply) = {{0, OUTPUT_DESC(z)}};

// TileD
INPUT_MAP(TileD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(TileD) = {{2, ATTR_DESC(multiples, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(TileD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TileD) = {{0, OUTPUT_DESC(y)}};

// OneHot
INPUT_MAP(OneHot) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(depth)}, {3, INPUT_DESC(on_value)}, {4, INPUT_DESC(off_value)}};
ATTR_MAP(OneHot) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};
OUTPUT_MAP(OneHot) = {{0, OUTPUT_DESC(y)}};

// GatherV2D
INPUT_MAP(GatherV2D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}};
INPUT_ATTR_MAP(GatherV2D) = {{3, ATTR_DESC(axis, AnyTraits<int64_t>())}};
ATTR_MAP(GatherV2D) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherV2D) = {{0, OUTPUT_DESC(y)}};

// Reshape
INPUT_MAP(Reshape) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(shape)}};
ATTR_MAP(Reshape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Reshape) = {{0, OUTPUT_DESC(y)}};

// BiasAdd
INPUT_MAP(BiasAdd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(bias)}};
ATTR_MAP(BiasAdd) = {{"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(BiasAdd) = {{0, OUTPUT_DESC(y)}};

// Iou
INPUT_MAP(Iou) = {{1, INPUT_DESC(bboxes)}, {2, INPUT_DESC(gtboxes)}};
ATTR_MAP(Iou) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(Iou) = {{0, OUTPUT_DESC(overlap)}};

// ResizeNearestNeighborV2D
INPUT_MAP(ResizeNearestNeighborV2D) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ResizeNearestNeighborV2D) = {
  {"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2D) = {{0, OUTPUT_DESC(y)}};

// ResizeNearestNeighborV2Grad
INPUT_MAP(ResizeNearestNeighborV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(size)}};
ATTR_MAP(ResizeNearestNeighborV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2Grad) = {{0, OUTPUT_DESC(y)}};

// ApplyAdam
INPUT_MAP(ApplyAdam) = {{1, INPUT_DESC(var)},         {2, INPUT_DESC(m)},           {3, INPUT_DESC(v)},
                        {4, INPUT_DESC(beta1_power)}, {5, INPUT_DESC(beta2_power)}, {6, INPUT_DESC(lr)},
                        {7, INPUT_DESC(beta1)},       {8, INPUT_DESC(beta2)},       {9, INPUT_DESC(epsilon)},
                        {10, INPUT_DESC(grad)}};
ATTR_MAP(ApplyAdam) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())},
                       {"use_nesterov", ATTR_DESC(use_nesterov, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyAdam) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(m)}, {2, OUTPUT_DESC(v)}};

// Relu6
INPUT_MAP(Relu6) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Relu6) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Relu6) = {{0, OUTPUT_DESC(y)}};

// Relu6Grad
INPUT_MAP(Relu6Grad) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(features)}};
ATTR_MAP(Relu6Grad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Relu6Grad) = {{0, OUTPUT_DESC(backprops)}};

// ResizeBilinearV2Grad
INPUT_MAP(ResizeBilinearV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(original_image)}};
ATTR_MAP(ResizeBilinearV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2Grad) = {{0, OUTPUT_DESC(y)}};

// ResizeBilinearV2D
INPUT_MAP(ResizeBilinearV2D) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ResizeBilinearV2D) = {
  {"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2D) = {{0, OUTPUT_DESC(y)}};

// ZerosLike
INPUT_MAP(ZerosLike) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ZerosLike) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ZerosLike) = {{0, OUTPUT_DESC(y)}};

// OnesLike
INPUT_MAP(OnesLike) = {{1, INPUT_DESC(x)}};
ATTR_MAP(OnesLike) = EMPTY_ATTR_MAP;
OUTPUT_MAP(OnesLike) = {{0, OUTPUT_DESC(y)}};

// NMSWithMask
INPUT_MAP(NMSWithMask) = {{1, INPUT_DESC(box_scores)}};
ATTR_MAP(NMSWithMask) = {{"iou_threshold", ATTR_DESC(iou_threshold, AnyTraits<float>())}};
OUTPUT_MAP(NMSWithMask) = {
  {0, OUTPUT_DESC(selected_boxes)}, {1, OUTPUT_DESC(selected_idx)}, {2, OUTPUT_DESC(selected_mask)}};

// Unpack
INPUT_MAP(Unpack) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Unpack) = {{"axis", ATTR_DESC(axis, AnyTraits<int>())}, {"num", ATTR_DESC(num, AnyTraits<int>())}};
DYN_OUTPUT_MAP(Unpack) = {{0, DYN_OUTPUT_DESC(y)}};

// ScatterNdUpdate
INPUT_MAP(ScatterNdUpdate) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterNdUpdate) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterNdUpdate) = {{0, OUTPUT_DESC(var)}};

// ScatterMax
INPUT_MAP(ScatterMax) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterMax) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ScatterMax) = {{0, OUTPUT_DESC(var)}};

// CheckValid
INPUT_MAP(CheckValid) = {{1, INPUT_DESC(bbox_tensor)}, {2, INPUT_DESC(img_metas)}};
ATTR_MAP(CheckValid) = EMPTY_ATTR_MAP;
OUTPUT_MAP(CheckValid) = {{0, OUTPUT_DESC(valid_tensor)}};

// SmoothL1Loss
INPUT_MAP(SmoothL1Loss) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(label)}};
ATTR_MAP(SmoothL1Loss) = {{"sigma", ATTR_DESC(sigma, AnyTraits<float>())}};
OUTPUT_MAP(SmoothL1Loss) = {{0, OUTPUT_DESC(loss)}};

// SmoothL1LossGrad
INPUT_MAP(SmoothL1LossGrad) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(label)}, {3, INPUT_DESC(dout)}};
ATTR_MAP(SmoothL1LossGrad) = {{"sigma", ATTR_DESC(sigma, AnyTraits<float>())}};
OUTPUT_MAP(SmoothL1LossGrad) = {{0, OUTPUT_DESC(gradient)}};

// SigmoidCrossEntropyWithLogits
INPUT_MAP(SigmoidCrossEntropyWithLogits) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(target)}};
ATTR_MAP(SigmoidCrossEntropyWithLogits) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SigmoidCrossEntropyWithLogits) = {{0, OUTPUT_DESC(loss)}};

// SigmoidCrossEntropyWithLogitsGrad
INPUT_MAP(SigmoidCrossEntropyWithLogitsGrad) = {
  {1, INPUT_DESC(predict)}, {2, INPUT_DESC(target)}, {3, INPUT_DESC(dout)}};
ATTR_MAP(SigmoidCrossEntropyWithLogitsGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SigmoidCrossEntropyWithLogitsGrad) = {{0, OUTPUT_DESC(gradient)}};

// ScatterNdD
INPUT_MAP(ScatterNdD) = {{1, INPUT_DESC(indices)}, {2, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ScatterNdD) = {
  {3, ATTR_DESC(shape, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ScatterNdD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ScatterNdD) = {{0, OUTPUT_DESC(y)}};

// PadD
INPUT_MAP(PadD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(PadD) = {{"paddings", ATTR_DESC(paddings, AnyTraits<std::vector<std::vector<int64_t>>>())}};
OUTPUT_MAP(PadD) = {{0, OUTPUT_DESC(y)}};

// MirrorPad
INPUT_MAP(MirrorPad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_MAP(MirrorPad) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(MirrorPad) = {{0, OUTPUT_DESC(y)}};

// MirrorPadGrad
INPUT_MAP(MirrorPadGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_MAP(MirrorPadGrad) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(MirrorPadGrad) = {{0, OUTPUT_DESC(y)}};

// GatherNd
INPUT_MAP(GatherNd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}};
ATTR_MAP(GatherNd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherNd) = {{0, OUTPUT_DESC(y)}};

// ROIAlign
INPUT_MAP(ROIAlign) = {{1, INPUT_DESC(features)}, {2, INPUT_DESC(rois)}};
OUTPUT_MAP(ROIAlign) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(ROIAlign) = {{"pooled_height", ATTR_DESC(pooled_height, AnyTraits<int>())},
                      {"pooled_width", ATTR_DESC(pooled_width, AnyTraits<int>())},
                      {"spatial_scale", ATTR_DESC(spatial_scale, AnyTraits<float>())},
                      {"sample_num", ATTR_DESC(sample_num, AnyTraits<int>())}};

// ROIAlignGrad
INPUT_MAP(ROIAlignGrad) = {{1, INPUT_DESC(ydiff)}, {2, INPUT_DESC(rois)}};
OUTPUT_MAP(ROIAlignGrad) = {{0, OUTPUT_DESC(xdiff)}};
ATTR_MAP(ROIAlignGrad) = {
  {"xdiff_shape", ATTR_DESC(xdiff_shape, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pooled_height", ATTR_DESC(pooled_height, AnyTraits<int>())},
  {"pooled_width", ATTR_DESC(pooled_width, AnyTraits<int>())},
  {"spatial_scale", ATTR_DESC(spatial_scale, AnyTraits<float>())},
  {"sample_num", ATTR_DESC(sample_num, AnyTraits<int>())}};

// ArgMaxD
INPUT_MAP(ArgMaxD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ArgMaxD) = {{"axis", ATTR_DESC(dimension, AnyTraits<int>())},
                     {"output_type", ATTR_DESC(dtype, AnyTraits<GEType>())}};
OUTPUT_MAP(ArgMaxD) = {{0, OUTPUT_DESC(y)}};

// ArgMinD
INPUT_MAP(ArgMinD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ArgMinD) = {{"axis", ATTR_DESC(dimension, AnyTraits<int>())},
                     {"output_type", ATTR_DESC(dtype, AnyTraits<GEType>())}};
OUTPUT_MAP(ArgMinD) = {{0, OUTPUT_DESC(y)}};

// ArgMaxWithValue
INPUT_MAP(ArgMaxWithValue) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ArgMaxWithValue) = {{"axis", ATTR_DESC(dimension, AnyTraits<int>())},
                             {"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ArgMaxWithValue) = {{0, OUTPUT_DESC(indice)}, {1, OUTPUT_DESC(values)}};

// ArgMinWithValue
INPUT_MAP(ArgMinWithValue) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ArgMinWithValue) = {{"axis", ATTR_DESC(dimension, AnyTraits<int>())},
                             {"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ArgMinWithValue) = {{0, OUTPUT_DESC(indice)}, {1, OUTPUT_DESC(values)}};

// ReduceAllD
INPUT_MAP(ReduceAllD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceAllD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceAllD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceAllD) = {{0, OUTPUT_DESC(y)}};

// ReduceMeanD
INPUT_MAP(ReduceMeanD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceMeanD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceMeanD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceMeanD) = {{0, OUTPUT_DESC(y)}};

// HCOMAllreduce
INPUT_MAP(HcomAllReduce) = {{1, INPUT_DESC(x)}};
OUTPUT_MAP(HcomAllReduce) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(HcomAllReduce) = {{"op", ATTR_DESC(reduction, AnyTraits<std::string>())},
                           {"group", ATTR_DESC(group, AnyTraits<std::string>())},
                           {"fusion", ATTR_DESC(fusion, AnyTraits<int>())}};

// HCOMBraodcast
INPUT_MAP(HcomBroadcast) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(HcomBroadcast) = {{1, DYN_INPUT_DESC(x)}};
DYN_OUTPUT_MAP(HcomBroadcast) = {{0, DYN_OUTPUT_DESC(y)}};
ATTR_MAP(HcomBroadcast) = {{"root_rank", ATTR_DESC(root_rank, AnyTraits<int>())},
                           {"group", ATTR_DESC(group, AnyTraits<std::string>())}};

// HCOMAllreduce
INPUT_MAP(HcomAllGather) = {{1, INPUT_DESC(x)}};
OUTPUT_MAP(HcomAllGather) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(HcomAllGather) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                           {"rank_size", ATTR_DESC(rank_size, AnyTraits<int>())}};

// HCOMReduceScatter
INPUT_MAP(HcomReduceScatter) = {{1, INPUT_DESC(x)}};
OUTPUT_MAP(HcomReduceScatter) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(HcomReduceScatter) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                               {"op", ATTR_DESC(reduction, AnyTraits<std::string>())},
                               {"rank_size", ATTR_DESC(rank_size, AnyTraits<int>())}};

// Variable
INPUT_MAP(Variable) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Variable) = EMPTY_ATTR_MAP;

// ReluGrad
INPUT_MAP(ReluGrad) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(features)}};
ATTR_MAP(ReluGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ReluGrad) = {{0, OUTPUT_DESC(backprops)}};

// BiasAddGrad
INPUT_MAP(BiasAddGrad) = {{1, INPUT_DESC(x)}};
ATTR_MAP(BiasAddGrad) = {{"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(BiasAddGrad) = {{0, OUTPUT_DESC(y)}};

// MaxPoolGrad
INPUT_MAP(MaxPoolGrad) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(grad)}};
ATTR_MAP(MaxPoolGrad) = {{"ksize", ATTR_DESC(ksize, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                         {"strides", ATTR_DESC(strides, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                         {"padding", ATTR_DESC(padding, AnyTraits<std::string>())},
                         {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPoolGrad) = {{0, OUTPUT_DESC(y)}};

// avgpoolgrad
INPUT_MAP(AvgPoolGrad) = {{1, INPUT_DESC(orig_input_shape)}, {2, INPUT_DESC(input_grad)}};
ATTR_MAP(AvgPoolGrad) = {{"ksize", ATTR_DESC(ksize, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                         {"strides", ATTR_DESC(strides, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                         {"padding", ATTR_DESC(padding, AnyTraits<std::string>())},
                         {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(AvgPoolGrad) = {{0, OUTPUT_DESC(out_grad)}};

// MaxPoolWithArgmax
INPUT_MAP(MaxPoolWithArgmax) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MaxPoolWithArgmax) = {{"ksize", ATTR_DESC(ksize, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                               {"strides", ATTR_DESC(strides, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                               {"padding", ATTR_DESC(padding, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPoolWithArgmax) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(argmax)}};

// MaxPoolGradWithArgmax
INPUT_MAP(MaxPoolGradWithArgmax) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grad)}, {3, INPUT_DESC(argmax)}};
ATTR_MAP(MaxPoolGradWithArgmax) = {{"ksize", ATTR_DESC(ksize, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                                   {"strides", ATTR_DESC(strides, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                                   {"padding", ATTR_DESC(padding, AnyTraits<std::string>())}};
OUTPUT_MAP(MaxPoolGradWithArgmax) = {{0, OUTPUT_DESC(y)}};

// ExtractImagePatches
INPUT_MAP(ExtractImagePatches) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ExtractImagePatches) = {{"ksizes", ATTR_DESC(ksizes, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                                 {"strides", ATTR_DESC(strides, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                                 {"rates", ATTR_DESC(rates, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())},
                                 {"padding", ATTR_DESC(padding, AnyTraits<std::string>())}};
OUTPUT_MAP(ExtractImagePatches) = {{0, OUTPUT_DESC(y)}};

// Conv2D
INPUT_MAP(Conv2D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}};
ATTR_MAP(Conv2D) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int>())},
};
OUTPUT_MAP(Conv2D) = {{0, OUTPUT_DESC(y)}};

// Conv2DBackpropInputD
INPUT_MAP(Conv2DBackpropInputD) = {{1, INPUT_DESC(out_backprop)}, {2, INPUT_DESC(filter)}};
INPUT_ATTR_MAP(Conv2DBackpropInputD) = {
  {3, ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(Conv2DBackpropInputD) = {
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"stride", ATTR_DESC(strides, "pad", AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int>())},
};
OUTPUT_MAP(Conv2DBackpropInputD) = {{0, OUTPUT_DESC(y)}};

// Conv2DBackpropFilterD
INPUT_MAP(Conv2DBackpropFilterD) = {{1, INPUT_DESC(out_backprop)}, {2, INPUT_DESC(x)}};
INPUT_ATTR_MAP(Conv2DBackpropFilterD) = {
  {3, ATTR_DESC(filter_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(Conv2DBackpropFilterD) = {
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"stride", ATTR_DESC(strides, "pad", AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int>())},
};
OUTPUT_MAP(Conv2DBackpropFilterD) = {{0, OUTPUT_DESC(y)}};

// DepthwiseConv2D
INPUT_MAP(DepthwiseConv2D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}};
ATTR_MAP(DepthwiseConv2D) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pads", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
};
OUTPUT_MAP(DepthwiseConv2D) = {{0, OUTPUT_DESC(y)}};

// DepthwiseConv2DBackpropInputD
INPUT_MAP(DepthwiseConv2DBackpropInputD) = {{2, INPUT_DESC(filter)}, {3, INPUT_DESC(out_backprop)}};
INPUT_ATTR_MAP(DepthwiseConv2DBackpropInputD) = {
  {1, ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(DepthwiseConv2DBackpropInputD) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pads", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
};
OUTPUT_MAP(DepthwiseConv2DBackpropInputD) = {{0, OUTPUT_DESC(input_grad)}};

// DepthwiseConv2DBackpropFilterD
INPUT_MAP(DepthwiseConv2DBackpropFilterD) = {{1, INPUT_DESC(input)}, {3, INPUT_DESC(out_backprop)}};
INPUT_ATTR_MAP(DepthwiseConv2DBackpropFilterD) = {
  {2, ATTR_DESC(filter_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(DepthwiseConv2DBackpropFilterD) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pads", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
};
OUTPUT_MAP(DepthwiseConv2DBackpropFilterD) = {{0, OUTPUT_DESC(filter_grad)}};

// MatMul
INPUT_MAP(MatMul) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(MatMul) = {{"transpose_a", ATTR_DESC(transpose_x1, AnyTraits<bool>())},
                    {"transpose_b", ATTR_DESC(transpose_x2, AnyTraits<bool>())}};
OUTPUT_MAP(MatMul) = {{0, OUTPUT_DESC(y)}};

// Merge
INPUT_MAP(Merge) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(Merge) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(Merge) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Merge) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(value_index)}};

// Switch
INPUT_MAP(Switch) = {{1, INPUT_DESC(data)}, {2, INPUT_DESC(pred)}};
OUTPUT_MAP(Switch) = {{0, OUTPUT_DESC(output_false)}, {1, OUTPUT_DESC(output_true)}};
ATTR_MAP(Switch) = EMPTY_ATTR_MAP;

// AddN
INPUT_MAP(AddN) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(AddN) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(AddN) = {{"n", ATTR_DESC(N, AnyTraits<int64_t>())}};
OUTPUT_MAP(AddN) = {{0, OUTPUT_DESC(y)}};

// Mul
INPUT_MAP(Mul) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Mul) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Mul) = {{0, OUTPUT_DESC(y)}};

// RealDiv
INPUT_MAP(RealDiv) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(RealDiv) = EMPTY_ATTR_MAP;
OUTPUT_MAP(RealDiv) = {{0, OUTPUT_DESC(y)}};

// Cast
INPUT_MAP(Cast) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(Cast) = {{2, ATTR_DESC(dst_type, AnyTraits<GEType>())}};
ATTR_MAP(Cast) = {{"Truncate", ATTR_DESC(truncate, AnyTraits<bool>())}};
OUTPUT_MAP(Cast) = {{0, OUTPUT_DESC(y)}};

// Reciprocal
INPUT_MAP(Reciprocal) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Reciprocal) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Reciprocal) = {{0, OUTPUT_DESC(y)}};

// Sub
INPUT_MAP(Sub) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Sub) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sub) = {{0, OUTPUT_DESC(y)}};

// SplitD
INPUT_MAP(SplitD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(SplitD) = {{"axis", ATTR_DESC(split_dim, AnyTraits<int>())},
                    {"output_num", ATTR_DESC(num_split, AnyTraits<int>())}};
DYN_OUTPUT_MAP(SplitD) = {{0, DYN_OUTPUT_DESC(y)}};

// Neg
INPUT_MAP(Neg) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Neg) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Neg) = {{0, OUTPUT_DESC(y)}};

// Transpose
INPUT_MAP(TransposeD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(TransposeD) = {{2, ATTR_DESC(perm, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(TransposeD) = EMPTY_ATTR_MAP;
// Do not set Transpose operator output descriptor

// DropOutGenMask
INPUT_MAP(DropOutGenMask) = {{1, INPUT_DESC(shape)}, {2, INPUT_DESC(prob)}};
ATTR_MAP(DropOutGenMask) = {{"Seed0", ATTR_DESC(seed, AnyTraits<int64_t>())},
                            {"Seed1", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(DropOutGenMask) = {{0, OUTPUT_DESC(y)}};

// Pack
INPUT_MAP(Pack) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(Pack) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(Pack) = {{"num", ATTR_DESC(N, AnyTraits<int>())}, {"axis", ATTR_DESC(axis, AnyTraits<int>())}};
OUTPUT_MAP(Pack) = {{0, OUTPUT_DESC(y)}};

// ConcatD
INPUT_MAP(ConcatD) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(ConcatD) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(ConcatD) = {
  {"axis", ATTR_DESC(concat_dim, AnyTraits<int>())},
  {"inputNums", ATTR_DESC(N, AnyTraits<int>())},
};
OUTPUT_MAP(ConcatD) = {{0, OUTPUT_DESC(y)}};

// Less
INPUT_MAP(Less) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Less) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Less) = {{0, OUTPUT_DESC(y)}};

// Rsqrt
INPUT_MAP(Rsqrt) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Rsqrt) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Rsqrt) = {{0, OUTPUT_DESC(y)}};

// Sqrt
INPUT_MAP(Sqrt) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Sqrt) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sqrt) = {{0, OUTPUT_DESC(y)}};

// Square
INPUT_MAP(Square) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Square) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Square) = {{0, OUTPUT_DESC(y)}};

// SquareSumAll
INPUT_MAP(SquareSumAll) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(SquareSumAll) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SquareSumAll) = {{0, OUTPUT_DESC(y1)}, {1, OUTPUT_DESC(y2)}};

// Tanh
INPUT_MAP(Tanh) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Tanh) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Tanh) = {{0, OUTPUT_DESC(y)}};

// TanhGrad
INPUT_MAP(TanhGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(TanhGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TanhGrad) = {{0, OUTPUT_DESC(z)}};

// ReduceMinD
INPUT_MAP(ReduceMinD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceMinD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceMinD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceMinD) = {{0, OUTPUT_DESC(y)}};

// ReduceMaxD
INPUT_MAP(ReduceMaxD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceMaxD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceMaxD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceMaxD) = {{0, OUTPUT_DESC(y)}};

// Maximum
INPUT_MAP(Maximum) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Maximum) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Maximum) = {{0, OUTPUT_DESC(y)}};

// Minimum
INPUT_MAP(Minimum) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Minimum) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Minimum) = {{0, OUTPUT_DESC(y)}};

// MaximumGrad
INPUT_MAP(MaximumGrad) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(grads)}};
ATTR_MAP(MaximumGrad) = {{"grad_x", ATTR_DESC(grad_x, AnyTraits<bool>())},
                         {"grad_y", ATTR_DESC(grad_y, AnyTraits<bool>())}};
OUTPUT_MAP(MaximumGrad) = {{0, OUTPUT_DESC(y1)}, {1, OUTPUT_DESC(y2)}};

// MinimumGrad
INPUT_MAP(MinimumGrad) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(grads)}};
ATTR_MAP(MinimumGrad) = {{"grad_x", ATTR_DESC(grad_x, AnyTraits<bool>())},
                         {"grad_y", ATTR_DESC(grad_y, AnyTraits<bool>())}};
OUTPUT_MAP(MinimumGrad) = {{0, OUTPUT_DESC(y1)}, {1, OUTPUT_DESC(y2)}};

// Pow
INPUT_MAP(Pow) = {
  {1, INPUT_DESC(x1)},
  {2, INPUT_DESC(x2)},
};
ATTR_MAP(Pow) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Pow) = {{0, OUTPUT_DESC(y)}};

// Equal
INPUT_MAP(Equal) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Equal) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Equal) = {{0, OUTPUT_DESC(y)}};

// NotEqual
INPUT_MAP(NotEqual) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(NotEqual) = EMPTY_ATTR_MAP;
OUTPUT_MAP(NotEqual) = {{0, OUTPUT_DESC(y)}};

// Log
INPUT_MAP(Log) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Log) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Log) = {{0, OUTPUT_DESC(y)}};

// LogicalAnd
INPUT_MAP(LogicalAnd) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(LogicalAnd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LogicalAnd) = {{0, OUTPUT_DESC(y)}};

// LogicalOr
INPUT_MAP(LogicalOr) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(LogicalOr) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LogicalOr) = {{0, OUTPUT_DESC(y)}};

// LogicalNot
INPUT_MAP(LogicalNot) = {{1, INPUT_DESC(x)}};
ATTR_MAP(LogicalNot) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LogicalNot) = {{0, OUTPUT_DESC(y)}};

// Greater
INPUT_MAP(Greater) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Greater) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Greater) = {{0, OUTPUT_DESC(y)}};

// LogSoftmaxGrad
INPUT_MAP(LogSoftmaxGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grad)}};
ATTR_MAP(LogSoftmaxGrad) = {
  {"axis", ATTR_DESC(axis, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(LogSoftmaxGrad) = {{0, OUTPUT_DESC(y)}};

// Select
INPUT_MAP(Select) = {{1, INPUT_DESC(condition)}, {2, INPUT_DESC(x1)}, {3, INPUT_DESC(x2)}};
ATTR_MAP(Select) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Select) = {{0, OUTPUT_DESC(y)}};

// LessEqual
INPUT_MAP(LessEqual) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(LessEqual) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LessEqual) = {{0, OUTPUT_DESC(y)}};

// LogSoftmaxV2
INPUT_MAP(LogSoftmaxV2) = {{1, INPUT_DESC(logits)}};
ATTR_MAP(LogSoftmaxV2) = {
  {"axis", ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(LogSoftmaxV2) = {{0, OUTPUT_DESC(logsoftmax)}};

// RandomChoiceWithMask
INPUT_MAP(RandomChoiceWithMask) = {{1, INPUT_DESC(x)}};
ATTR_MAP(RandomChoiceWithMask) = {{"count", ATTR_DESC(count, AnyTraits<int64_t>())},
                                  {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                  {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(RandomChoiceWithMask) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mask)}};

// TruncatedNormal
INPUT_MAP(TruncatedNormal) = {{1, INPUT_DESC(shape)}};
ATTR_MAP(TruncatedNormal) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                             {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(TruncatedNormal) = {{0, OUTPUT_DESC(y)}};

// StridedSliceGrad
INPUT_MAP(StridedSliceGrad) = {
  {1, INPUT_DESC(dy)}, {2, INPUT_DESC(shape)}, {3, INPUT_DESC(begin)}, {4, INPUT_DESC(end)}, {5, INPUT_DESC(strides)}};
ATTR_MAP(StridedSliceGrad) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                              {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                              {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                              {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                              {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};
OUTPUT_MAP(StridedSliceGrad) = {{0, OUTPUT_DESC(output)}};

// Gelu
INPUT_MAP(Gelu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Gelu) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Gelu) = {{0, OUTPUT_DESC(y)}};

// GeluGrad
INPUT_MAP(GeluGrad) = {{1, INPUT_DESC(dy)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(y)}};
ATTR_MAP(GeluGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GeluGrad) = {{0, OUTPUT_DESC(z)}};

// StridedSlice
INPUT_MAP(StridedSlice) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(begin)}, {3, INPUT_DESC(end)}, {4, INPUT_DESC(strides)}};
ATTR_MAP(StridedSlice) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                          {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                          {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                          {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                          {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};
OUTPUT_MAP(StridedSlice) = {{0, OUTPUT_DESC(y)}};

// UnsortedSegmentSum
INPUT_MAP(UnsortedSegmentSumD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}};
INPUT_ATTR_MAP(UnsortedSegmentSumD) = {{3, ATTR_DESC(num_segments, AnyTraits<int64_t>())}};
ATTR_MAP(UnsortedSegmentSumD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnsortedSegmentSumD) = {{0, OUTPUT_DESC(y)}};

// ExpandDims
INPUT_MAP(ExpandDims) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axis)}};
ATTR_MAP(ExpandDims) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ExpandDims) = {{0, OUTPUT_DESC(y)}};

// Squeeze
INPUT_MAP(Squeeze) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Squeeze) = {{"axis", ATTR_DESC(axis, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Squeeze) = {{0, OUTPUT_DESC(y)}};

// SGD
INPUT_MAP(SGD) = {{1, INPUT_DESC(parameters)}, {2, INPUT_DESC(gradient)}, {3, INPUT_DESC(learning_rate)},
                  {4, INPUT_DESC(accum)},      {5, INPUT_DESC(momentum)}, {6, INPUT_DESC(stat)}};
ATTR_MAP(SGD) = {{"dampening", ATTR_DESC(dampening, AnyTraits<float>())},
                 {"weight_decay", ATTR_DESC(weight_decay, AnyTraits<float>())},
                 {"nesterov", ATTR_DESC(nesterov, AnyTraits<bool>())}};
OUTPUT_MAP(SGD) = {{0, OUTPUT_DESC(parameters)}};

// LayerNorm
INPUT_MAP(LayerNorm) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(gamma)}, {3, INPUT_DESC(beta)}};
ATTR_MAP(LayerNorm) = {{"begin_norm_axis", ATTR_DESC(begin_norm_axis, AnyTraits<int>())},
                       {"begin_params_axis", ATTR_DESC(begin_params_axis, AnyTraits<int>())}};
OUTPUT_MAP(LayerNorm) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mean)}, {2, OUTPUT_DESC(variance)}};

// LayerNormGrad
INPUT_MAP(LayerNormGrad) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(dy)}, {3, INPUT_DESC(variance)}, {4, INPUT_DESC(mean)}, {5, INPUT_DESC(gamma)}};
ATTR_MAP(LayerNormGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LayerNormGrad) = {{0, OUTPUT_DESC(pd_x)}, {1, OUTPUT_DESC(pd_gamma)}, {2, OUTPUT_DESC(pd_beta)}};

// BatchMatMul
INPUT_MAP(BatchMatMul) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(BatchMatMul) = {{"transpose_x1", ATTR_DESC(adj_x1, AnyTraits<bool>())},
                         {"transpose_x2", ATTR_DESC(adj_x2, AnyTraits<bool>())}};
OUTPUT_MAP(BatchMatMul) = {{0, OUTPUT_DESC(y)}};

// DropoutDoMask
INPUT_MAP(DropOutDoMask) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(mask)}, {3, INPUT_DESC(keep_prob)}};
ATTR_MAP(DropOutDoMask) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DropOutDoMask) = {{0, OUTPUT_DESC(y)}};

// NPUGetFloatStatus
INPUT_MAP(NPUGetFloatStatus) = {{1, INPUT_DESC(addr)}};
OUTPUT_MAP(NPUGetFloatStatus) = {{0, OUTPUT_DESC(data)}};
ATTR_MAP(NPUGetFloatStatus) = EMPTY_ATTR_MAP;

// NPUAllocFloatStatus
INPUT_MAP(NPUAllocFloatStatus) = EMPTY_INPUT_MAP;
ATTR_MAP(NPUAllocFloatStatus) = EMPTY_ATTR_MAP;
OUTPUT_MAP(NPUAllocFloatStatus) = {{0, OUTPUT_DESC(data)}};

// NPUClearFloatStatus
INPUT_MAP(NPUClearFloatStatus) = {{1, INPUT_DESC(addr)}};
OUTPUT_MAP(NPUClearFloatStatus) = {{0, OUTPUT_DESC(data)}};
ATTR_MAP(NPUClearFloatStatus) = EMPTY_ATTR_MAP;

// Abs
INPUT_MAP(Abs) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Abs) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Abs) = {{0, OUTPUT_DESC(y)}};

// AbsGrad
INPUT_MAP(AbsGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(AbsGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AbsGrad) = {{0, OUTPUT_DESC(z)}};

// BinaryCrossEntropy
INPUT_MAP(BinaryCrossEntropy) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(weight)}};
ATTR_MAP(BinaryCrossEntropy) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(BinaryCrossEntropy) = {{0, OUTPUT_DESC(output)}};

// BinaryCrossEntropyGrad
INPUT_MAP(BinaryCrossEntropyGrad) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(grad_output)}, {4, INPUT_DESC(weight)}};
ATTR_MAP(BinaryCrossEntropyGrad) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(BinaryCrossEntropyGrad) = {{0, OUTPUT_DESC(output)}};

// SparseApplyAdagradD
INPUT_MAP(SparseApplyAdagradD) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(grad)}, {4, INPUT_DESC(indices)}};
ATTR_MAP(SparseApplyAdagradD) = {{"lr", ATTR_DESC(lr, AnyTraits<float>())},
                                 {"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(SparseApplyAdagradD) = {{0, OUTPUT_DESC(var)}};

// SparseApplyFtrlD
INPUT_MAP(SparseApplyFtrlD) = {{1, INPUT_DESC(var)},
                               {2, INPUT_DESC(accum)},
                               {3, INPUT_DESC(linear)},
                               {4, INPUT_DESC(grad)},
                               {5, INPUT_DESC(indices)}};
ATTR_MAP(SparseApplyFtrlD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())},
                              {"lr", ATTR_DESC(lr, AnyTraits<float>())},
                              {"l1", ATTR_DESC(l1, AnyTraits<float>())},
                              {"l2", ATTR_DESC(l2, AnyTraits<float>())},
                              {"lr_power", ATTR_DESC(lr_power, AnyTraits<float>())}};
OUTPUT_MAP(SparseApplyFtrlD) = {{0, OUTPUT_DESC(var)}};

// SpaceToDepth
INPUT_MAP(SpaceToDepth) = {{1, INPUT_DESC(x)}};
ATTR_MAP(SpaceToDepth) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())}};
OUTPUT_MAP(SpaceToDepth) = {{0, OUTPUT_DESC(y)}};

// DepthToSpace
INPUT_MAP(DepthToSpace) = {{1, INPUT_DESC(x)}};
ATTR_MAP(DepthToSpace) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())}};
OUTPUT_MAP(DepthToSpace) = {{0, OUTPUT_DESC(y)}};

// Sign
INPUT_MAP(Sign) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Sign) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sign) = {{0, OUTPUT_DESC(y)}};

// Round
INPUT_MAP(Round) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Round) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Round) = {{0, OUTPUT_DESC(y)}};

// ApplyFtrl
INPUT_MAP(ApplyFtrl) = {{1, INPUT_DESC(var)},  {2, INPUT_DESC(accum)},   {3, INPUT_DESC(linear)},
                        {4, INPUT_DESC(grad)}, {5, INPUT_DESC(lr)},      {6, INPUT_DESC(l1)},
                        {7, INPUT_DESC(l2)},   {8, INPUT_DESC(lr_power)}};
ATTR_MAP(ApplyFtrl) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyFtrl) = {{0, OUTPUT_DESC(var)}};

// Diag
INPUT_MAP(Diag) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Diag) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Diag) = {{0, OUTPUT_DESC(y)}};

// DiagPart
INPUT_MAP(DiagPart) = {{1, INPUT_DESC(x)}};
ATTR_MAP(DiagPart) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DiagPart) = {{0, OUTPUT_DESC(y)}};

// SpaceToBatchD
INPUT_MAP(SpaceToBatchD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(SpaceToBatchD) = {
  {"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
  {"paddings", ATTR_DESC(paddings, AnyTraits<std::vector<std::vector<int64_t>>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(SpaceToBatchD) = {{0, OUTPUT_DESC(y)}};

// BatchToSpaceD
INPUT_MAP(BatchToSpaceD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(BatchToSpaceD) = {
  {"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
  {"crops", ATTR_DESC(crops, AnyTraits<std::vector<std::vector<int64_t>>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(BatchToSpaceD) = {{0, OUTPUT_DESC(y)}};

// Atan2
INPUT_MAP(Atan2) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Atan2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Atan2) = {{0, OUTPUT_DESC(y)}};

// ApplyRMSPropD
INPUT_MAP(ApplyRMSPropD) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(ms)}, {3, INPUT_DESC(mom)}, {4, INPUT_DESC(grad)}, {5, INPUT_DESC(lr)}};
INPUT_ATTR_MAP(ApplyRMSPropD) = {{6, ATTR_DESC(rho, AnyTraits<float>())},
                                 {7, ATTR_DESC(momentum, AnyTraits<float>())},
                                 {8, ATTR_DESC(epsilon, AnyTraits<float>())}};
ATTR_MAP(ApplyRMSPropD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyRMSPropD) = {{0, OUTPUT_DESC(var)}};

// ApplyCenteredRMSProp
INPUT_MAP(ApplyCenteredRMSProp) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(mg)},       {3, INPUT_DESC(ms)},
                                   {4, INPUT_DESC(mom)}, {5, INPUT_DESC(grad)},     {6, INPUT_DESC(lr)},
                                   {7, INPUT_DESC(rho)}, {8, INPUT_DESC(momentum)}, {9, INPUT_DESC(epsilon)}};
ATTR_MAP(ApplyCenteredRMSProp) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyCenteredRMSProp) = {{0, OUTPUT_DESC(var)}};

// L2Loss
INPUT_MAP(L2Loss) = {{1, INPUT_DESC(x)}};
ATTR_MAP(L2Loss) = EMPTY_ATTR_MAP;
OUTPUT_MAP(L2Loss) = {{0, OUTPUT_DESC(y)}};

// CTCLoss
INPUT_MAP(CTCLoss) = {{1, INPUT_DESC(inputs)},
                      {2, INPUT_DESC(labels_indices)},
                      {3, INPUT_DESC(labels_values)},
                      {4, INPUT_DESC(sequence_length)}};
ATTR_MAP(CTCLoss) = {
  {"preprocess_collapse_repeated", ATTR_DESC(preprocess_collapse_repeated, AnyTraits<bool>())},
  {"ctc_merge_repeated", ATTR_DESC(ctc_merge_repeated, AnyTraits<bool>())},
  {"ignore_longer_outputs_than_inputs", ATTR_DESC(ignore_longer_outputs_than_inputs, AnyTraits<bool>())}};
OUTPUT_MAP(CTCLoss) = {{0, OUTPUT_DESC(loss)}, {1, OUTPUT_DESC(gradient)}};

#ifdef ENABLE_GE
// Print
INPUT_MAP(Print) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(Print) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(Print) = EMPTY_ATTR_MAP;
#endif
}  // namespace transform
}  // namespace mindspore
