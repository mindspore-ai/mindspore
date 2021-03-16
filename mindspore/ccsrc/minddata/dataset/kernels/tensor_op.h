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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TENSOR_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TENSOR_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/core/device_resource.h"

#define IO_CHECK(input, output)                             \
  do {                                                      \
    if (input == nullptr || output == nullptr) {            \
      RETURN_STATUS_UNEXPECTED("input or output is null."); \
    }                                                       \
  } while (false)

#define IO_CHECK_VECTOR(input, output)              \
  do {                                              \
    if (output == nullptr) {                        \
      RETURN_STATUS_UNEXPECTED("output is null.");  \
    }                                               \
    for (auto &_i : input) {                        \
      if (_i == nullptr) {                          \
        RETURN_STATUS_UNEXPECTED("input is null."); \
      }                                             \
    }                                               \
  } while (false)

namespace mindspore {
namespace dataset {

// base class
constexpr char kTensorOp[] = "TensorOp";

// image
constexpr char kAffineOp[] = "AffineOp";
constexpr char kAutoContrastOp[] = "AutoContrastOp";
constexpr char kBoundingBoxAugmentOp[] = "BoundingBoxAugmentOp";
constexpr char kDecodeOp[] = "DecodeOp";
constexpr char kCenterCropOp[] = "CenterCropOp";
constexpr char kCutMixBatchOp[] = "CutMixBatchOp";
constexpr char kCutOutOp[] = "CutOutOp";
constexpr char kCropOp[] = "CropOp";
constexpr char kDvppCropJpegOp[] = "DvppCropJpegOp";
constexpr char kDvppDecodeResizeCropJpegOp[] = "DvppDecodeResizeCropJpegOp";
constexpr char kDvppDecodeResizeJpegOp[] = "DvppDecodeResizeJpegOp";
constexpr char kDvppDecodeJpegOp[] = "DvppDecodeJpegOp";
constexpr char kDvppDecodePngOp[] = "DvppDecodePngOp";
constexpr char kDvppNormalizeOp[] = "DvppNormalizeOp";
constexpr char kDvppResizeJpegOp[] = "DvppResizeJpegOp";
constexpr char kEqualizeOp[] = "EqualizeOp";
constexpr char kHwcToChwOp[] = "HWC2CHWOp";
constexpr char kInvertOp[] = "InvertOp";
constexpr char kMixUpBatchOp[] = "MixUpBatchOp";
constexpr char kNormalizeOp[] = "NormalizeOp";
constexpr char kNormalizePadOp[] = "NormalizePadOp";
constexpr char kPadOp[] = "PadOp";
constexpr char kRandomAffineOp[] = "RandomAffineOp";
constexpr char kRandomColorAdjustOp[] = "RandomColorAdjustOp";
constexpr char kRandomCropAndResizeOp[] = "RandomCropAndResizeOp";
constexpr char kRandomCropAndResizeWithBBoxOp[] = "RandomCropAndResizeWithBBoxOp";
constexpr char kRandomCropDecodeResizeOp[] = "RandomCropDecodeResizeOp";
constexpr char kRandomCropOp[] = "RandomCropOp";
constexpr char kRandomCropWithBBoxOp[] = "RandomCropWithBBoxOp";
constexpr char kRandomHorizontalFlipWithBBoxOp[] = "RandomHorizontalFlipWithBBoxOp";
constexpr char kRandomHorizontalFlipOp[] = "RandomHorizontalFlipOp";
constexpr char kRandomResizeOp[] = "RandomResizeOp";
constexpr char kRandomResizeWithBBoxOp[] = "RandomResizeWithBBoxOp";
constexpr char kRandomRotationOp[] = "RandomRotationOp";
constexpr char kRandomSolarizeOp[] = "RandomSolarizeOp";
constexpr char kRandomSharpnessOp[] = "RandomSharpnessOp";
constexpr char kRandomVerticalFlipOp[] = "RandomVerticalFlipOp";
constexpr char kRandomVerticalFlipWithBBoxOp[] = "RandomVerticalFlipWithBBoxOp";
constexpr char kRescaleOp[] = "RescaleOp";
constexpr char kResizeBilinearOp[] = "ResizeBilinearOp";
constexpr char kResizeOp[] = "ResizeOp";
constexpr char kResizePreserveAROp[] = "ResizePreserveAROp";
constexpr char kResizeWithBBoxOp[] = "ResizeWithBBoxOp";
constexpr char kRgbaToBgrOp[] = "RgbaToBgrOp";
constexpr char kRgbaToRgbOp[] = "RgbaToRgbOp";
constexpr char kRgbToGrayOp[] = "RgbToGrayOp";
constexpr char kSharpnessOp[] = "SharpnessOp";
constexpr char kSolarizeOp[] = "SolarizeOp";
constexpr char kSwapRedBlueOp[] = "SwapRedBlueOp";
constexpr char kUniformAugOp[] = "UniformAugOp";
constexpr char kSoftDvppDecodeRandomCropResizeJpegOp[] = "SoftDvppDecodeRandomCropResizeJpegOp";
constexpr char kSoftDvppDecodeReiszeJpegOp[] = "SoftDvppDecodeReiszeJpegOp";
constexpr char kRandomColorOp[] = "RandomColorOp";

// text
constexpr char kBasicTokenizerOp[] = "BasicTokenizerOp";
constexpr char kBertTokenizerOp[] = "BertTokenizerOp";
constexpr char kCaseFoldOp[] = "CaseFoldOp";
constexpr char kJiebaTokenizerOp[] = "JiebaTokenizerOp";
constexpr char kLookupOp[] = "LookupOp";
constexpr char kNgramOp[] = "NgramOp";
constexpr char kSlidingWindowOp[] = "SlidingWindowOp";
constexpr char kNormalizeUTF8Op[] = "NormalizeUTF8Op";
constexpr char kRegexReplaceOp[] = "RegexReplaceOp";
constexpr char kRegexTokenizerOp[] = "RegexTokenizerOp";
constexpr char kToNumberOp[] = "ToNumberOp";
constexpr char kTruncateSequencePairOp[] = "TruncateSequencePairOp";
constexpr char kUnicodeCharTokenizerOp[] = "UnicodeCharTokenizerOp";
constexpr char kUnicodeScriptTokenizerOp[] = "UnicodeScriptTokenizerOp";
constexpr char kWhitespaceTokenizerOp[] = "WhitespaceTokenizerOp";
constexpr char kWordpieceTokenizerOp[] = "WordpieceTokenizerOp";
constexpr char kRandomChoiceOp[] = "RandomChoiceOp";
constexpr char kRandomApplyOp[] = "RandomApplyOp";
constexpr char kComposeOp[] = "Compose";
constexpr char kRandomSelectSubpolicyOp[] = "RandomSelectSubpolicyOp";
constexpr char kSentencepieceTokenizerOp[] = "SentencepieceTokenizerOp";

// data
constexpr char kConcatenateOp[] = "ConcatenateOp";
constexpr char kDuplicateOp[] = "DuplicateOp";
constexpr char kFillOp[] = "FillOp";
constexpr char kMaskOp[] = "MaskOp";
constexpr char kOneHotOp[] = "OneHotOp";
constexpr char kPadEndOp[] = "PadEndOp";
constexpr char kSliceOp[] = "SliceOp";
constexpr char kToFloat16Op[] = "ToFloat16Op";
constexpr char kTypeCastOp[] = "TypeCastOp";
constexpr char kUniqueOp[] = "UniqueOp";

// other
constexpr char kCFuncOp[] = "CFuncOp";
constexpr char kPyFuncOp[] = "PyFuncOp";
constexpr char kNoOp[] = "NoOp";

// A class that does a computation on a Tensor
class TensorOp {
 public:
  TensorOp() = default;

  virtual ~TensorOp() = default;

  // A function that prints info about the tensor operation
  // @param out
  virtual void Print(std::ostream &out) const { out << Name() << std::endl; }

  // Provide stream operator for displaying it
  // @param output stream
  // @param so the TensorOp object to be printed
  // @return output stream
  friend std::ostream &operator<<(std::ostream &out, const TensorOp &so) {
    so.Print(out);
    return out;
  }

  // Perform an operation on one Tensor and produce one Tensor. This is for 1-to-1 column MapOp
  // @param input  shares the ownership of the Tensor (increase the ref count).
  // @param output the address to a shared_ptr where the result will be placed.
  // @return Status
  virtual Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

  // Perform an operation on Tensors from multiple columns, and produce multiple Tensors.
  // This is for m-to-n column MapOp.
  // @param input is a vector of shared_ptr to Tensor (pass by const reference).
  // @param output is the address to an empty vector of shared_ptr to Tensor.
  // @return Status
  virtual Status Compute(const TensorRow &input, TensorRow *output);

  // Perform an operation on one DeviceTensor and produce one DeviceTensor. This is for 1-to-1 column MapOp
  // @param input shares the ownership of the Tensor (increase the ref count).
  // @param output the address to a shared_ptr where the result will be placed.
  // @return Status
  virtual Status Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output);

  // Returns true oif the TensorOp takes one input and returns one output.
  // @return true/false
  bool OneToOne() { return NumInput() == 1 && NumOutput() == 1; }

  // Returns true oif the TensorOp produces deterministic result.
  // @return true/false
  bool Deterministic() { return is_deterministic_; }

  // Function to determine the number of inputs the TensorOp can take. 0: means undefined.
  // @return uint32_t
  virtual uint32_t NumInput() { return 1; }

  // Function to determine the number of output the TensorOp generates. 0: means undefined.
  // @return uint32_t
  virtual uint32_t NumOutput() { return 1; }

  // Function to determine the shapes of the output tensor given the input tensors' shapes.
  // If a subclass did not override this function, it means that the shape does not change.
  // @param inputs in: vector of the shapes of the input tensors.
  // @param outputs out: vector of the shapes of the output tensors to be filled.
  // @return Status
  virtual Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs);

  // Function to determine the types of the output tensor given the input tensor's types.
  // If a subclass did not override this function, it means that the type does not change.
  // @param inputs in: vector of the types of the input tensors.
  // @param outputs out: vector of the types of the output tensors to be filled.
  // @return Status
  virtual Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs);

  virtual std::string Name() const = 0;

  virtual Status to_json(nlohmann::json *out_json) { return Status::OK(); }

  virtual Status SetAscendResource(const std::shared_ptr<DeviceResource> &resource);

 protected:
  bool is_deterministic_{true};
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TENSOR_OP_H_
