/*
# Adapted from
# https://github.com/tensorflow/tensorflow/blob/v2.4.2/tensorflow/core/util/example_proto_fast_parsing.cc
# Additional modifications are made by Huawei Technologies Co., Ltd in 2024.

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "minddata/dataset/general/kernels/parse_example_op.h"

#include <google/protobuf/io/coded_stream.h>

#include <algorithm>
#include <memory>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "proto/example.pb.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/general/kernels/data_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/thread_bind_core_utils.h"

namespace mindspore::dataset {
namespace protobuf = ::google::protobuf;

constexpr bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
constexpr size_t kInlinedVectorSize = 4;

template <typename T>
using SmallVector = absl::InlinedVector<T, kInlinedVectorSize>;
using StringPiece = std::string_view;

template <typename T>
class LimitedArraySlice {
 public:
  using value_type = T;

  LimitedArraySlice(T *begin, size_t num_elements) : current_(begin), begin_(begin), end_(begin + num_elements) {}

  /// \brief Get the left space in the slice.
  int64_t EndDistance() const { return end_ - current_; }

  /// \brief Push value to back of slice. If the slice is full, only change the
  /// total number without modify the data.
  void push_back(T &&value) {
    if (EndDistance() > 0) {
      *current_ = std::move(value);
    }
    ++current_;
  }

  /// \brief Construct an element at the back of slice and return a mutable
  /// reference to the new element.
  T &construct_at_end() {
    if (EndDistance() <= 0) {
      MS_EXCEPTION(RuntimeError) << "LimitedArraySlice has no space left.";
    }
    return *(current_++);
  }

  /// \brief Get the mutable reference to the last element in slice.
  T &back() { return *(current_ - 1); }

  /// \brief Get the number of elements in slice.
  size_t size() const { return std::min(current_ - begin_, end_ - begin_); }

  /// \brief Resize the slice to the given size by advancing the pointer to
  /// the current element.
  void resize(size_t size) { current_ = begin_ + size; }

  /// \brief Get the data buffer.
  T *data() { return begin_; }

 private:
  T *current_;
  T *begin_;
  T *end_;
};

uint8_t PeekTag(protobuf::io::CodedInputStream *stream) {
  if (stream == nullptr) {
    MS_EXCEPTION(RuntimeError) << "CodedInputStream is nullptr.";
  }
  const void *ptr;
  int size;
  if (!stream->GetDirectBufferPointer(&ptr, &size)) {
    return 0;
  }
  if (ptr == nullptr) {
    return 0;
  }
  return *static_cast<const uint8_t *>(ptr);
}

constexpr uint8_t kVarintTag(const uint32_t tag) { return (tag << 3) | 0; }
constexpr uint8_t kDelimitedTag(const uint32_t tag) { return (tag << 3) | 2; }
constexpr uint8_t kFixed32Tag(const uint32_t tag) { return (tag << 3) | 5; }

namespace parsed {
class Feature {
 public:
  Feature() = default;
  explicit Feature(const StringPiece &serialized) : serialized_(serialized) {}

  Status ParseDataType(DataType *dtype) {
    RETURN_UNEXPECTED_IF_NULL(dtype);
    if (serialized_.empty()) {
      *dtype = DataType(DataType::DE_UNKNOWN);
      return Status::OK();
    }
    const auto oneof_tag = static_cast<uint8_t>(*serialized_.data());
    serialized_.remove_prefix(1);
    constexpr uint8_t kStringTag = 1;
    constexpr uint8_t kFloat32Tag = 2;
    constexpr uint8_t kInt64Tag = 3;
    switch (oneof_tag) {
      case kDelimitedTag(kStringTag):
        *dtype = DataType(DataType::DE_STRING);
        break;
      case kDelimitedTag(kFloat32Tag):
        *dtype = DataType(DataType::DE_FLOAT32);
        break;
      case kDelimitedTag(kInt64Tag):
        *dtype = DataType(DataType::DE_INT64);
        break;
      default:
        // Initialize variable to avoid compiler warning
        *dtype = DataType(DataType::DE_UNKNOWN);
        RETURN_STATUS_UNEXPECTED("Unsupported datatype.");
    }
    return Status::OK();
  }

  bool GetNumElementsInBytesList(int *num_elements) const {
    if (num_elements == nullptr) {
      return false;
    }
    protobuf::io::CodedInputStream stream(reinterpret_cast<const uint8_t *>(serialized_.data()),
                                          static_cast<int>(serialized_.size()));
    uint32_t length = 0;
    if (!stream.ReadVarint32(&length)) {
      return false;
    }
    const auto limit = stream.PushLimit(static_cast<int>(length));
    *num_elements = 0;
    while (!stream.ExpectAtEnd()) {
      if (!stream.ExpectTag(kDelimitedTag(1))) {
        return false;
      }
      uint32_t bytes_length = 0;
      if (!stream.ReadVarint32(&bytes_length)) {
        return false;
      }
      if (!stream.Skip(static_cast<int>(bytes_length))) {
        return false;
      }
      ++*num_elements;
    }
    stream.PopLimit(limit);
    return true;
  }

  static std::string *construct_at_end(LimitedArraySlice<std::string> *bytes_list) {
    if (bytes_list == nullptr) {
      MS_EXCEPTION(RuntimeError) << "LimitedArraySlice is nullptr.";
    }
    if (bytes_list->EndDistance() <= 0) {
      return nullptr;
    }
    return &bytes_list->construct_at_end();
  }

  static std::string *construct_at_end(std::vector<std::string> *bytes_list) {
    if (bytes_list == nullptr) {
      MS_EXCEPTION(RuntimeError) << "String vector is nullptr.";
    }
    return &bytes_list->emplace_back();
  }

  template <typename Result>
  bool ParseBytesList(Result *bytes_list) const {
    if (bytes_list == nullptr) {
      return false;
    }

    protobuf::io::CodedInputStream stream(reinterpret_cast<const uint8_t *>(serialized_.data()),
                                          static_cast<int>(serialized_.size()));

    uint32_t length;
    if (!stream.ReadVarint32(&length)) {
      return false;
    }
    const auto limit = stream.PushLimit(static_cast<int>(length));

    while (!stream.ExpectAtEnd()) {
      if (!stream.ExpectTag(kDelimitedTag(1))) {
        return false;
      }
      // parse string
      uint32_t bytes_length;
      if (!stream.ReadVarint32(&bytes_length)) {
        return false;
      }
      std::string *bytes = construct_at_end(bytes_list);
      if (bytes == nullptr) {
        return false;
      }
      bytes->resize(bytes_length);
      if (!stream.ReadRaw(bytes->data(), static_cast<int>(bytes_length))) {
        return false;
      }
    }
    stream.PopLimit(limit);
    return true;
  }

  template <typename Result>
  bool ParseFloatList(Result *float_list) const {
    if (float_list == nullptr) {
      return false;
    }
    protobuf::io::CodedInputStream stream(reinterpret_cast<const uint8_t *>(serialized_.data()),
                                          static_cast<int>(serialized_.size()));
    uint32_t length;
    if (!stream.ReadVarint32(&length)) {
      return false;
    }
    const auto limit = stream.PushLimit(static_cast<int>(length));

    if (!stream.ExpectAtEnd()) {
      const uint8_t peek_tag = PeekTag(&stream);
      if (peek_tag != kDelimitedTag(1) && peek_tag != kFixed32Tag(1)) {
        return false;
      }

      constexpr int32_t kNumFloatBytes = 4;
      if (peek_tag == kDelimitedTag(1)) {           // packed
        if (!stream.ExpectTag(kDelimitedTag(1))) {  // packed tag
          return false;
        }
        uint32_t packed_length;
        if (!stream.ReadVarint32(&packed_length)) {
          return false;
        }
        const auto packed_limit = stream.PushLimit(static_cast<int>(packed_length));

        // Store the initial size to know the offset we have to start writing
        // data from before resizing the output "vector".
        const size_t initial_size = float_list->size();
        float_list->resize(initial_size + packed_length / kNumFloatBytes);

        // If the result data type is float and we are on a little endian
        // machine then we can simply memcpy the data from the proto into the
        // result vector.
        if (kLittleEndian && sizeof(typename Result::value_type) == kNumFloatBytes) {
          // Calculate the length of the buffer available what can be less than
          // what we requested in resize in case of a LimitedArraySlice.
          const uint32_t bytes_to_copy =
            std::min(static_cast<uint32_t>((float_list->size() - initial_size) * kNumFloatBytes), packed_length);
          if (!stream.ReadRaw(float_list->data() + initial_size, bytes_to_copy)) {
            return false;
          }
        } else {
          int64_t index = initial_size;
          while (!stream.ExpectAtEnd()) {
            uint32_t buffer32;
            if (!stream.ReadLittleEndian32(&buffer32)) {
              return false;
            }
            if (index < float_list->size()) {
              float_list->data()[index] = absl::bit_cast<float>(buffer32);
              ++index;
            }
          }
        }

        stream.PopLimit(packed_limit);
      } else {  // non-packed
        const size_t initial_size = float_list->size();
        // 1 byte for the tag (`1` encoded as Variant32) and kNumFloatBytes for
        // the value.
        const int64_t num_elements = stream.BytesUntilLimit() / (1 + kNumFloatBytes);
        float_list->resize(initial_size + num_elements);
        int64_t index = initial_size;
        while (!stream.ExpectAtEnd()) {
          if (!stream.ExpectTag(kFixed32Tag(1))) {
            return false;
          }
          uint32_t buffer32;
          if (!stream.ReadLittleEndian32(&buffer32)) {
            return false;
          }
          float_list->data()[index] = absl::bit_cast<float>(buffer32);
          ++index;
        }
      }
    }

    stream.PopLimit(limit);
    return true;
  }

  template <typename Result>
  bool ParseInt64List(Result *int64_list) const {
    if (int64_list == nullptr) {
      return false;
    }
    protobuf::io::CodedInputStream stream(reinterpret_cast<const uint8_t *>(serialized_.data()),
                                          static_cast<int>(serialized_.size()));
    uint32_t length;
    if (!stream.ReadVarint32(&length)) {
      return false;
    }
    const auto limit = stream.PushLimit(static_cast<int>(length));

    if (!stream.ExpectAtEnd()) {
      const uint8_t peek_tag = PeekTag(&stream);
      if (peek_tag != kDelimitedTag(1) && peek_tag != kVarintTag(1)) {
        return false;
      }
      if (peek_tag == kDelimitedTag(1)) {           // packed
        if (!stream.ExpectTag(kDelimitedTag(1))) {  // packed tag
          return false;
        }
        uint32_t packed_length;
        if (!stream.ReadVarint32(&packed_length)) {
          return false;
        }
        const auto packed_limit = stream.PushLimit(static_cast<int>(packed_length));

        while (!stream.ExpectAtEnd()) {
          uint64_t n;  // There is no API for int64
          if (!stream.ReadVarint64(&n)) {
            return false;
          }
          int64_list->push_back(static_cast<int64_t>(n));
        }

        stream.PopLimit(packed_limit);
      } else {  // non-packed
        while (!stream.ExpectAtEnd()) {
          if (!stream.ExpectTag(kVarintTag(1))) {
            return false;
          }
          uint64_t n;  // There is no API for int64
          if (!stream.ReadVarint64(&n)) {
            return false;
          }
          int64_list->push_back(static_cast<int64_t>(n));
        }
      }
    }
    stream.PopLimit(limit);
    return true;
  }

 private:
  StringPiece serialized_;
};

using FeatureMapEntry = std::pair<StringPiece, Feature>;
using Example = std::vector<FeatureMapEntry>;
}  // namespace parsed

inline bool SkipExtraneousTag(protobuf::io::CodedInputStream *stream) {
  if (stream == nullptr) {
    MS_EXCEPTION(RuntimeError) << "CodedInputStream is nullptr.";
  }
  uint32_t data;
  uint64_t dummy;
  constexpr uint32_t kVarint = 0;
  constexpr uint32_t kFixed64 = 1;
  constexpr uint32_t kLengthDelimited = 2;
  constexpr uint32_t kGroupBegin = 3;
  constexpr uint32_t kGroupEnd = 4;
  constexpr uint32_t kFixed32 = 5;
  switch (stream->ReadTag() & 0x7) {
    case kVarint:  // varint
      return stream->ReadVarint32(&data);
    case kFixed64:  // fixed64
      return stream->ReadLittleEndian64(&dummy);
    case kLengthDelimited:  // length delimited
      if (!stream->ReadVarint32(&data)) {
        return false;
      }
      stream->Skip(static_cast<int>(data));
      return true;
    case kGroupBegin:  // group begin
    case kGroupEnd:    // group end
      return false;    // groups not supported.
    case kFixed32:     // fixed32
      return stream->ReadLittleEndian32(&data);
    default:
      return false;
  }
  return false;  // unrecognized tag type
}

bool ParseString(protobuf::io::CodedInputStream *stream, StringPiece *result) {
  if (stream == nullptr) {
    return false;
  }
  if (result == nullptr) {
    return false;
  }
  uint32_t length;
  if (!stream->ReadVarint32(&length)) {
    return false;
  }
  if (length == 0) {
    *result = StringPiece(nullptr, 0);
    return true;
  }
  const void *stream_alias;
  int stream_size;
  if (!stream->GetDirectBufferPointer(&stream_alias, &stream_size)) {
    return false;
  }
  if (static_cast<uint32_t>(stream_size) < length) {
    return false;
  }
  *result = StringPiece(static_cast<const char *>(stream_alias), length);
  stream->Skip(static_cast<int>(length));
  return true;
}

bool ParseFeatureMapEntry(protobuf::io::CodedInputStream *stream, parsed::FeatureMapEntry *feature_map_entry) {
  if (stream == nullptr) {
    return false;
  }
  if (feature_map_entry == nullptr) {
    return false;
  }
  uint32_t length;
  if (!stream->ReadVarint32(&length)) {
    return false;
  }
  const auto limit = stream->PushLimit(static_cast<int>(length));

  // Protobufs allow an arbitrary order for the key and value fields.
  for (int n = 0; n <= 1; ++n) {
    constexpr uint32_t kNameTag = 1;
    constexpr uint32_t kFeatureTag = 2;
    switch (stream->ReadTag()) {
      case kDelimitedTag(kNameTag):
        if (!ParseString(stream, &feature_map_entry->first)) {
          return false;
        }
        break;

      case kDelimitedTag(kFeatureTag): {
        StringPiece feature_string_piece;
        if (!ParseString(stream, &feature_string_piece)) {
          return false;
        }
        feature_map_entry->second = parsed::Feature(feature_string_piece);
        break;
      }

      default:
        return false;
    }
  }

  if (!stream->ExpectAtEnd()) {
    return false;
  }
  stream->PopLimit(limit);
  return true;
}

bool ParseFeatures(protobuf::io::CodedInputStream *stream, parsed::Example *example) {
  if (stream == nullptr) {
    return false;
  }
  if (example == nullptr) {
    return false;
  }
  uint32_t length;
  if (!stream->ReadVarint32(&length)) {
    return false;
  }
  const auto limit = stream->PushLimit(static_cast<int>(length));
  while (!stream->ExpectAtEnd()) {
    parsed::FeatureMapEntry feature_map_entry;
    if (!stream->ExpectTag(kDelimitedTag(1))) {
      return false;
    }
    if (!ParseFeatureMapEntry(stream, &feature_map_entry)) {
      return false;
    }
    example->push_back(std::move(feature_map_entry));
  }
  stream->PopLimit(limit);
  return true;
}

bool ParseExample(protobuf::io::CodedInputStream *stream, parsed::Example *example) {
  if (stream == nullptr) {
    return false;
  }
  if (example == nullptr) {
    return false;
  }
  // Loop over the input stream which may contain multiple serialized Example
  // protos merged together as strings. This behavior is consistent with Proto's
  // ParseFromString when string representations are concatenated.
  while (!stream->ExpectAtEnd()) {
    if (!stream->ExpectTag(kDelimitedTag(1))) {
      if (!SkipExtraneousTag(stream)) {
        return false;
      }
    } else {
      if (!ParseFeatures(stream, example)) {
        return false;
      }
    }
  }
  return true;
}

bool ParseExample(const StringPiece &serialized, parsed::Example *example) {
  if (example == nullptr) {
    return false;
  }
  protobuf::io::CodedInputStream stream(reinterpret_cast<const uint8_t *>(serialized.data()),
                                        static_cast<int>(serialized.size()));
  return ParseExample(&stream, example);
}

template <typename T>
class TensorVector {
 public:
  using value_type = T;

  std::shared_ptr<Tensor> tensor() {
    if (tensor_ == nullptr) {
      resize(0);
    }
    return tensor_;
  }

  int64_t size() const { return tensor_ != nullptr ? tensor_->Size() : 0; }

  void resize(int64_t new_size) {
    if (tensor_ != nullptr) {
      MS_EXCEPTION(RuntimeError) << "TensorVector has already initialized.";
    }
    Status s = Tensor::CreateEmpty(TensorShape({new_size}), DataType::FromCType<T>(), &tensor_);
    if (s.IsError()) {
      MS_EXCEPTION(RuntimeError) << s.ToString();
    }
    data_ = &*(tensor_->begin<T>());
  }

  T *data() { return data_; }

  const T *data() const { return data_; }

 private:
  std::shared_ptr<Tensor> tensor_ = nullptr;
  T *data_ = nullptr;  // the raw data inside the tensor
};

template <typename T>
void CopyOrMoveBlock(const T *b, const T *e, T *t) {
  std::copy(b, e, t);
}

void LogFeatureRepeated(const StringPiece &feature_name) {
  MS_LOG(WARNING) << "Feature name: " << feature_name << " is repeated in Example. Ignoring all but last one.";
}

inline Status ReportUnexpectedParseFailure(const StringPiece &feature_name) {
  RETURN_STATUS_UNEXPECTED("Failed to parse serialized Example of feature name: " + std::string(feature_name));
}

inline Status ReportUnexpectedDataType(const StringPiece &feature_name, const DataType &dtype) {
  RETURN_STATUS_UNEXPECTED("Got unexpected data type: " + dtype.ToString() +
                           " of feature name: " + std::string(feature_name));
}

inline Status ReportUnexpectedDataShape(const StringPiece &feature_name) {
  RETURN_STATUS_UNEXPECTED("Column shape of " + std::string(feature_name) +
                           " defined in schema does not match the shape actually load.");
}

Status CreateUint8TensorFromString(const std::vector<std::string> &bytes_list, std::shared_ptr<Tensor> *column_tensor,
                                   const TensorShape &column_shape, const std::string &column_name) {
  RETURN_UNEXPECTED_IF_NULL(column_tensor);
  dsize_t total_size =
    std::accumulate(bytes_list.begin(), bytes_list.end(), 0,
                    [](dsize_t size, const std::string &str) { return size + static_cast<dsize_t>(str.size()); });
  TensorShape output_shape = column_shape;
  if (!column_shape.known()) {
    output_shape = TensorShape({total_size});
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(
      output_shape.NumOfElements() == total_size,
      "Column shape of " + column_name + " defined in schema does not match the shape actually load.");
  }
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(output_shape, DataType(DataType::DE_UINT8), column_tensor));
  ptrdiff_t offset = 0;
  for (const auto &str : bytes_list) {
    int ret_code = memcpy_s((*column_tensor)->GetMutableBuffer() + offset, (*column_tensor)->SizeInBytes() - offset,
                            common::SafeCStr(str), str.size());
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK, "Failed to copy string into Tensor.");
    offset += static_cast<ptrdiff_t>(str.size());
  }
  return Status::OK();
}

Status ParseExampleOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  if (parallel_parse_) {
    return ParallelParseExample(input, output);
  } else {
    return ParseSingleExample(input, output);
  }
}

Status ParseSingleKnownShapeColumn(const parsed::Feature &feature, std::shared_ptr<Tensor> *column_tensor,
                                   const StringPiece &feature_name, const ColDescriptor &column_descriptor,
                                   const DataType &example_dtype) {
  RETURN_UNEXPECTED_IF_NULL(column_tensor);
  const size_t num_elements = column_descriptor.Shape().NumOfElements();
  switch (example_dtype.value()) {
    case DataType::DE_INT64: {
      const auto data_buffer = reinterpret_cast<int64_t *>((*column_tensor)->GetMutableBuffer());
      LimitedArraySlice<int64_t> slice(data_buffer, num_elements);
      if (!feature.ParseInt64List(&slice)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      if (slice.EndDistance() != 0) {
        return ReportUnexpectedDataShape(feature_name);
      }
      break;
    }
    case DataType::DE_FLOAT32: {
      const auto data_buffer = reinterpret_cast<float *>((*column_tensor)->GetMutableBuffer());
      LimitedArraySlice<float> slice(data_buffer, num_elements);
      if (!feature.ParseFloatList(&slice)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      if (slice.EndDistance() != 0) {
        return ReportUnexpectedDataShape(feature_name);
      }
      break;
    }
    case DataType::DE_STRING: {
      std::vector<std::string> bytes_list;
      bytes_list.reserve(num_elements);
      if (!feature.ParseBytesList(&bytes_list)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      if (column_descriptor.Type().value() == DataType::DE_STRING) {
        if (bytes_list.size() != num_elements) {
          return ReportUnexpectedDataShape(feature_name);
        }
        TensorShape string_tensor_shape = TensorShape::CreateUnknownRankShape();
        RETURN_IF_NOT_OK(column_descriptor.MaterializeTensorShape(num_elements, &string_tensor_shape));
        RETURN_IF_NOT_OK(
          Tensor::CreateFromVector(bytes_list, string_tensor_shape, DataType(DataType::DE_STRING), column_tensor));
      } else {
        // load string or bytes as uint8 tensor
        RETURN_IF_NOT_OK(
          CreateUint8TensorFromString(bytes_list, column_tensor, column_descriptor.Shape(), std::string(feature_name)));
      }
      break;
    }
    default:
      return ReportUnexpectedDataType(feature_name, example_dtype);
  }
  return Status::OK();
}

Status ParseSingleVarLenColumn(const parsed::Feature &feature, std::shared_ptr<Tensor> *column_tensor,
                               const StringPiece &feature_name, const ColDescriptor &column_descriptor,
                               const DataType &example_dtype) {
  RETURN_UNEXPECTED_IF_NULL(column_tensor);
  std::vector<std::string> bytes_list;
  TensorVector<float> float_list;
  SmallVector<int64_t> int64_list;

  size_t num_elements;
  switch (example_dtype.value()) {
    case DataType::DE_INT64: {
      if (!feature.ParseInt64List(&int64_list)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      num_elements = int64_list.size();
      break;
    }
    case DataType::DE_FLOAT32: {
      if (!feature.ParseFloatList(&float_list)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      num_elements = float_list.size();
      break;
    }
    case DataType::DE_STRING: {
      int actual_num_elements = 0;
      if (!feature.GetNumElementsInBytesList(&actual_num_elements)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      bytes_list.reserve(actual_num_elements);
      if (!feature.ParseBytesList(&bytes_list)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      num_elements = bytes_list.size();
      break;
    }
    default:
      return ReportUnexpectedDataType(feature_name, example_dtype);
  }

  TensorShape column_shape = TensorShape::CreateUnknownRankShape();
  RETURN_IF_NOT_OK(column_descriptor.MaterializeTensorShape(num_elements, &column_shape));

  switch (example_dtype.value()) {
    case DataType::DE_INT64: {
      RETURN_IF_NOT_OK(Tensor::CreateEmpty(column_shape, example_dtype, column_tensor));
      CopyOrMoveBlock(int64_list.begin(), int64_list.end(),
                      reinterpret_cast<int64_t *>((*column_tensor)->GetMutableBuffer()));
      break;
    }
    case DataType::DE_FLOAT32: {
      RETURN_IF_NOT_OK(Tensor::CreateFromTensor(std::shared_ptr<Tensor>(float_list.tensor()), column_tensor));
      RETURN_IF_NOT_OK((*column_tensor)->Reshape(column_shape));
      break;
    }
    case DataType::DE_STRING: {
      if (column_descriptor.Type().value() == DataType::DE_STRING) {
        RETURN_IF_NOT_OK(
          Tensor::CreateFromVector(bytes_list, column_shape, DataType(DataType::DE_STRING), column_tensor));
      } else {
        // load string or bytes as uint8 tensor
        RETURN_IF_NOT_OK(CreateUint8TensorFromString(bytes_list, column_tensor, TensorShape::CreateUnknownRankShape(),
                                                     std::string(feature_name)));
      }
      break;
    }
    default:
      return ReportUnexpectedDataType(feature_name, example_dtype);
  }
  return Status::OK();
}

Status ParseExampleOp::ParseSingleExample(const TensorRow &raw_bytes, TensorRow *parsed_row) {
  RETURN_UNEXPECTED_IF_NULL(parsed_row);
  const auto filename = raw_bytes.getPath().empty() ? "" : raw_bytes.getPath()[0];
  const auto tensor_iterator = raw_bytes[0]->begin<std::string_view>();

  const auto example_bytes = std::string(*tensor_iterator);
  RETURN_IF_NOT_OK(ConstructColumnMap(example_bytes));

  parsed::Example parsed_example;
  CHECK_FAIL_RETURN_UNEXPECTED(ParseExample(example_bytes, &parsed_example),
                               "Failed to parse example bytes: " + example_bytes + " in tfrecord file: " + filename);

  parsed_row->reserve(data_schema_.NumColumns());

  for (int32_t column_index = 0; column_index < data_schema_.NumColumns(); ++column_index) {
    const ColDescriptor &column_descriptor = data_schema_.Column(column_index);
    if (column_descriptor.HasKnownShape()) {
      if (!column_descriptor.Type().IsString()) {
        DataType type;
        if (column_descriptor.Type().IsInt() || column_descriptor.Type().IsBool()) {
          type = DataType(DataType::DE_INT64);
        } else if (column_descriptor.Type().IsFloat()) {
          type = DataType(DataType::DE_FLOAT32);
        }
        std::shared_ptr<Tensor> column_tensor;
        RETURN_IF_NOT_OK(Tensor::CreateEmpty(column_descriptor.Shape(), type, &column_tensor));
        parsed_row->emplace_back(std::move(column_tensor));
      } else {
        parsed_row->emplace_back(std::make_shared<Tensor>(TensorShape({}), DataType(DataType::DE_UNKNOWN)));
      }
    } else {
      MS_LOG(INFO) << "Shape of column name: " << column_descriptor.Name() << " is not defined.";
      parsed_row->emplace_back(std::make_shared<Tensor>(TensorShape({}), DataType(DataType::DE_UNKNOWN)));
    }
  }

  std::vector<bool> feature_already_seen(data_schema_.NumColumns(), false);
  std::vector<std::string> file_paths;

  const size_t parsed_example_size = parsed_example.size();
  for (size_t i = 0; i < parsed_example_size; ++i) {
    // This is a logic that standard protobuf parsing is implementing.
    // I.e. last entry in the map overwrites all the previous ones.
    parsed::FeatureMapEntry &name_and_feature = parsed_example[parsed_example_size - i - 1];

    const StringPiece &feature_name = name_and_feature.first;
    parsed::Feature &feature = name_and_feature.second;

    if (column_name_id_map_.find(std::string(feature_name)) == column_name_id_map_.end()) {
      MS_LOG(INFO) << "Feature name: " << feature_name << " is not in schema, skip it.";
      continue;
    }

    const auto column_index = column_name_id_map_[std::string(feature_name)];

    DataType example_dtype;
    RETURN_IF_NOT_OK(feature.ParseDataType(&example_dtype));
    if (example_dtype == DataType::DE_UNKNOWN) {
      continue;
    }

    // If feature was already visited, skip.
    if (feature_already_seen[column_index]) {
      LogFeatureRepeated(feature_name);
      continue;
    }
    feature_already_seen[column_index] = true;

    const ColDescriptor &column_descriptor = data_schema_.Column(column_index);
    bool type_cast_flag = false;
    if (example_dtype != column_descriptor.Type()) {
      const std::string msg =
        "The data type loaded from the example for feature name: " + column_descriptor.Name() +
        " does not match the predefined type in schema, the actual type: " + example_dtype.ToString() +
        ", but the predefined type: " + column_descriptor.Type().ToString();
      if (!example_dtype.IsString() && !column_descriptor.Type().IsString()) {
        MS_LOG(INFO) << msg << ". This will cause a type cast.";
        type_cast_flag = true;
      } else if (column_descriptor.Type().value() != DataType::DE_UINT8) {
        // allow to read data of type string or bytes into an uint8 tensor
        RETURN_STATUS_UNEXPECTED(msg);
      }
    }

    if (column_descriptor.HasKnownShape()) {
      RETURN_IF_NOT_OK(ParseSingleKnownShapeColumn(feature, &(*parsed_row)[column_index], feature_name,
                                                   column_descriptor, example_dtype));
    } else {  // if variable length
      RETURN_IF_NOT_OK(
        ParseSingleVarLenColumn(feature, &(*parsed_row)[column_index], feature_name, column_descriptor, example_dtype));
    }
    if (type_cast_flag) {
      std::shared_ptr<Tensor> cast_out;
      RETURN_IF_NOT_OK(TypeCast((*parsed_row)[column_index], &cast_out, column_descriptor.Type()));
      (*parsed_row)[column_index] = cast_out;
    }
    file_paths.push_back(filename);
  }

  for (int32_t column_index = 0; column_index < data_schema_.NumColumns(); ++column_index) {
    CHECK_FAIL_RETURN_UNEXPECTED(feature_already_seen[column_index],
                                 "Feature name: " + data_schema_.Column(column_index).Name() +
                                   " is required in schema but could not be found in tfrecord file.");
  }

  parsed_row->setPath(file_paths);
  return Status::OK();
}

size_t CalculateNumMiniBatch(const std::shared_ptr<Tensor> &batch_tensor) {
  // This parameter affects performance in a big and data-dependent way.
  constexpr size_t kMiniBatchSizeBytes = 50000;

  const size_t batch_size = batch_tensor->shape()[0];

  size_t result = 0;
  size_t minibatch_bytes = 0;
  for (size_t i = 0; i < batch_size; i++) {
    if (minibatch_bytes == 0) {  // start minibatch
      result++;
    }
    std::string_view tensor_value;
    batch_tensor->GetItemAt(&tensor_value, {static_cast<dsize_t>(i)});
    minibatch_bytes += tensor_value.size() + 1;
    if (minibatch_bytes > kMiniBatchSizeBytes) {
      minibatch_bytes = 0;
    }
  }
  // 'special logic'
  const size_t min_minibatches = std::min<size_t>(8, batch_size);
  constexpr size_t max_minibatches = 64;
  return std::max<size_t>(min_minibatches, std::min<size_t>(max_minibatches, result));
}

class BlockingCounter {
 public:
  explicit BlockingCounter(const uint32_t initial_count) : state_(initial_count << 1), notified_(false) {
    if ((initial_count << 1) >> 1 != initial_count) {
      MS_EXCEPTION(RuntimeError) << "Value of initial_count exceeds upper limit: " << initial_count;
    }
  }

  ~BlockingCounter() = default;

  inline void DecrementCount() {
    constexpr uint32_t kStep = 2;
    uint32_t new_state = state_.fetch_sub(kStep, std::memory_order_acq_rel) - kStep;
    if (new_state != 1) {
      if (((new_state + kStep) & ~1) == 0) {
        MS_EXCEPTION(RuntimeError) << "The number of remaining worker threads is already 0.";
      }
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (notified_) {
      MS_EXCEPTION(RuntimeError) << "Try to awake a notified worker.";
    }
    notified_ = true;
    cond_var_.notify_all();
  }

  inline void Wait() {
    uint32_t new_state = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((new_state >> 1) == 0) {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    while (!notified_) {
      cond_var_.wait(lock);
    }
  }

  // Wait for the specified time, return false iff the count has not dropped to
  // zero before the timeout expired.
  inline bool WaitFor(std::chrono::milliseconds millisecond) {
    uint32_t new_state = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((new_state >> 1) == 0) {
      return true;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    while (!notified_) {
      const std::cv_status status = cond_var_.wait_for(lock, millisecond);
      if (status == std::cv_status::timeout) {
        return false;
      }
    }
    return true;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cond_var_;
  std::atomic<uint32_t> state_;  // low bit is waiter flag
  bool notified_;
};

void ParallelFor(const std::function<void(size_t)> &function, const size_t task_count,
                 const std::unique_ptr<Eigen::ThreadPool> &thread_pool) {
  if (task_count == 0) {
    return;
  }
  if (thread_pool == nullptr) {
    for (size_t i = 0; i < task_count; ++i) {
      function(i);
    }
  } else {
    BlockingCounter counter(task_count - 1);
    for (size_t i = 1; i < task_count; ++i) {
      thread_pool->Schedule([i, &function, &counter] {
        function(i);
        counter.DecrementCount();
      });
    }
    function(0);
    counter.Wait();
  }
}

Status FillAndCopyVarLenTensor(const std::vector<std::vector<VarLenTensorBuffer>> &minibatch_row_buffer,
                               std::shared_ptr<Tensor> *column_tensor, const size_t column_index) {
  RETURN_UNEXPECTED_IF_NULL(column_tensor);
  ptrdiff_t buffer_offset = 0;
  for (const auto &minibatch_row : minibatch_row_buffer) {
    const auto &minibatch_tensor = minibatch_row[column_index].numeric_tensor;
    for (const auto &varlen_tensor : minibatch_tensor) {
      const auto tensor_buffer_size = varlen_tensor->SizeInBytes();
      const errno_t copy_status =
        memcpy_s((*column_tensor)->GetMutableBuffer() + buffer_offset, (*column_tensor)->SizeInBytes() - buffer_offset,
                 varlen_tensor->GetBuffer(), tensor_buffer_size);
      CHECK_FAIL_RETURN_UNEXPECTED(copy_status == EOK,
                                   "Failed to copy tensor to batch, got error_t: " + std::to_string(copy_status));
      buffer_offset += tensor_buffer_size;
    }
  }
  return Status::OK();
}

Status FillAndCopyVarLenString(const std::vector<std::vector<VarLenTensorBuffer>> &minibatch_row_buffer,
                               std::shared_ptr<Tensor> *column_tensor, const size_t column_index,
                               const ColDescriptor &column_descriptor, dsize_t batch_size) {
  RETURN_UNEXPECTED_IF_NULL(column_tensor);
  std::vector<std::string> string_buffer;
  dsize_t element_size = 0;
  for (const auto &minibatch_row : minibatch_row_buffer) {
    const auto string_length = minibatch_row[column_index].string_length;
    if (element_size == 0) {
      element_size = static_cast<dsize_t>(string_length);
    } else {
      CHECK_FAIL_RETURN_UNEXPECTED(string_length == element_size,
                                   "Could not batch string or bytes tensors with different shapes.");
    }
    const auto &minibatch_string = minibatch_row[column_index].string_tensor;
    string_buffer.insert(string_buffer.end(), minibatch_string.begin(), minibatch_string.end());
  }

  std::vector<dsize_t> shape;
  if (element_size != 0) {
    shape = {batch_size, element_size};
  } else {
    shape = {batch_size};
  }
  const auto column_shape = TensorShape(shape);
  if (column_descriptor.Type().value() == DataType::DE_STRING) {
    RETURN_IF_NOT_OK(
      Tensor::CreateFromVector(string_buffer, column_shape, DataType(DataType::DE_STRING), column_tensor));
  } else {
    RETURN_IF_NOT_OK(CreateUint8TensorFromString(string_buffer, column_tensor, column_shape, column_descriptor.Name()));
  }
  return Status::OK();
}

Status MergeDenseVarLenMiniBatches(const std::vector<std::vector<VarLenTensorBuffer>> &varlen_dense_buffers,
                                   TensorRow *parsed_row, int32_t column_index, const DataSchema &data_schema,
                                   dsize_t batch_size) {
  RETURN_UNEXPECTED_IF_NULL(parsed_row);
  const ColDescriptor &column_descriptor = data_schema.Column(column_index);
  if (column_descriptor.HasKnownShape()) {
    return Status::OK();
  }
  std::shared_ptr<Tensor> column_tensor;
  if (!varlen_dense_buffers[0][column_index].numeric_tensor.empty()) {
    const TensorShape column_shape =
      varlen_dense_buffers[0][column_index].numeric_tensor[0]->shape().InsertDim(0, batch_size);
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(column_shape, column_descriptor.Type(), &column_tensor));
    RETURN_IF_NOT_OK(FillAndCopyVarLenTensor(varlen_dense_buffers, &column_tensor, column_index));
  } else {
    RETURN_IF_NOT_OK(
      FillAndCopyVarLenString(varlen_dense_buffers, &column_tensor, column_index, column_descriptor, batch_size));
  }
  (*parsed_row)[column_index] = column_tensor;
  return Status::OK();
}

Status ParseExampleOp::ParallelParseExample(const TensorRow &raw_bytes, TensorRow *parsed_row) {
  RETURN_UNEXPECTED_IF_NULL(parsed_row);
  Tensor::TensorIterator tensor_iterator = raw_bytes[0]->begin<std::string_view>();
  RETURN_IF_NOT_OK(ConstructColumnMap(std::string(*tensor_iterator)));
  parsed_row->reserve(data_schema_.NumColumns());

  auto batch_size = raw_bytes[0]->shape()[0];
  std::vector<bool> type_cast_flag(data_schema_.NumColumns(), false);
  std::vector<bool> varlen_column(data_schema_.NumColumns(), false);
  std::unordered_map<int32_t, std::vector<std::string>> string_column_map;
  for (int32_t column_index = 0; column_index < data_schema_.NumColumns(); ++column_index) {
    const ColDescriptor &column_descriptor = data_schema_.Column(column_index);
    if (column_descriptor.HasKnownShape()) {
      if (!column_descriptor.Type().IsString()) {
        auto column_shape = column_descriptor.Shape().InsertDim(0, batch_size);
        DataType type;
        if (column_descriptor.Type().IsInt() || column_descriptor.Type().IsBool()) {
          if (column_descriptor.Type().value() != DataType::DE_INT64) {
            type_cast_flag[column_index] = true;
          }
          type = DataType(DataType::DE_INT64);
        } else if (column_descriptor.Type().IsFloat()) {
          if (column_descriptor.Type().value() != DataType::DE_FLOAT32) {
            type_cast_flag[column_index] = true;
          }
          type = DataType(DataType::DE_FLOAT32);
        }
        std::shared_ptr<Tensor> column_tensor;
        RETURN_IF_NOT_OK(Tensor::CreateEmpty(column_shape, type, &column_tensor));
        parsed_row->emplace_back(std::move(column_tensor));
        if (column_descriptor.Type().value() == DataType::DE_UINT8) {
          string_column_map[column_index] =
            std::vector<std::string>(batch_size * column_descriptor.Shape().NumOfElements());
        }
      } else {
        parsed_row->emplace_back(std::make_shared<Tensor>(TensorShape({}), DataType(DataType::DE_UNKNOWN)));
        string_column_map[column_index] =
          std::vector<std::string>(batch_size * column_descriptor.Shape().NumOfElements());
      }
    } else {
      MS_LOG(INFO) << "Shape of column name: " << column_descriptor.Name() << " is not defined.";
      varlen_column[column_index] = true;
      parsed_row->emplace_back(std::make_shared<Tensor>(TensorShape({}), DataType(DataType::DE_UNKNOWN)));
    }
  }

  // Calculate number of minibatches.
  // In main regime make each minibatch around kMiniBatchSizeBytes bytes.
  // Apply 'special logic' below for small and big regimes.
  const size_t num_minibatches = CalculateNumMiniBatch(raw_bytes[0]);

  auto first_example_of_minibatch = [&](const size_t minibatch) -> size_t {
    return (batch_size * minibatch) / num_minibatches;
  };

  std::vector<std::vector<VarLenTensorBuffer>> varlen_dense_buffers(num_minibatches);
  std::vector<Status> status_of_minibatch(num_minibatches);
  auto ProcessMiniBatch = [&](const size_t minibatch) {
    mindspore::dataset::BindThreadCoreForMindDataOp("dataset::ParseExampleOp::ParallelParseExample::ProcessMiniBatch");
    varlen_dense_buffers[minibatch].resize(data_schema_.NumColumns());
    const auto start = first_example_of_minibatch(minibatch);
    const auto end = first_example_of_minibatch(minibatch + 1);
    for (auto tensor_index = start; tensor_index < end; ++tensor_index) {
      status_of_minibatch[minibatch] =
        ParseSerializedExample(static_cast<std::string>(*tensor_iterator.operator+(static_cast<dsize_t>(tensor_index))),
                               parsed_row, &string_column_map, &varlen_dense_buffers[minibatch], tensor_index);
      if (!status_of_minibatch[minibatch].IsOk()) {
        break;
      }
    }
  };

  CheckAndInitPool();

  ParallelFor(ProcessMiniBatch, num_minibatches, pool_);

  for (Status &status : status_of_minibatch) {
    RETURN_IF_NOT_OK(status);
  }

  for (auto string_column = string_column_map.begin(); string_column != string_column_map.end(); ++string_column) {
    auto column_index = string_column->first;
    const ColDescriptor &column_descriptor = data_schema_.Column(column_index);
    auto column_shape = column_descriptor.Shape().InsertDim(0, batch_size);
    std::shared_ptr<Tensor> string_tensor;
    if (column_descriptor.Type().value() == DataType::DE_STRING) {
      RETURN_IF_NOT_OK(
        Tensor::CreateFromVector(string_column->second, column_shape, DataType(DataType::DE_STRING), &string_tensor));
    } else {
      // load string or bytes as uint8 tensor
      RETURN_IF_NOT_OK(
        CreateUint8TensorFromString(string_column->second, &string_tensor, column_shape, column_descriptor.Name()));
      type_cast_flag[column_index] = false;
    }
    (*parsed_row)[column_index] = string_tensor;
  }

  for (int32_t column_index = 0; column_index < data_schema_.NumColumns(); ++column_index) {
    if (type_cast_flag[column_index]) {
      const ColDescriptor &column_descriptor = data_schema_.Column(column_index);
      std::shared_ptr<Tensor> cast_out;
      RETURN_IF_NOT_OK(TypeCast((*parsed_row)[column_index], &cast_out, column_descriptor.Type()));
      (*parsed_row)[column_index] = cast_out;
    } else if (varlen_column[column_index]) {
      RETURN_IF_NOT_OK(
        MergeDenseVarLenMiniBatches(varlen_dense_buffers, parsed_row, column_index, data_schema_, batch_size));
    }
  }
  return Status::OK();
}

Status ParseSerializedKnownShapeColumn(const parsed::Feature &feature, TensorRow *parsed_row,
                                       std::unordered_map<int32_t, std::vector<std::string>> *string_col_map,
                                       const int32_t column_index, const size_t tensor_index,
                                       const StringPiece &feature_name, const ColDescriptor &column_descriptor,
                                       const DataType &example_dtype) {
  RETURN_UNEXPECTED_IF_NULL(parsed_row);
  RETURN_UNEXPECTED_IF_NULL(string_col_map);
  std::shared_ptr<Tensor> &column_tensor = (*parsed_row)[column_index];
  if (example_dtype != column_descriptor.Type()) {
    const std::string msg =
      "The data type loaded from the example for feature name: " + column_descriptor.Name() +
      " does not match the predefined type in schema, the actual type: " + example_dtype.ToString() +
      ", but the predefined type: " + column_descriptor.Type().ToString();
    if (example_dtype == column_tensor->type()) {
      // if the actual data type is the same as the pre-allocated tensor,
      // we can first read it into the tensor, then cast to the type specified by the schema
      MS_LOG(INFO) << msg << ". This will cause a type cast.";
    } else if (!example_dtype.IsString() || column_descriptor.Type().value() != DataType::DE_UINT8) {
      // allow to read data of type string or bytes into an uint8 tensor
      RETURN_STATUS_UNEXPECTED(msg);
    }
  }

  const std::size_t num_elements = column_descriptor.Shape().NumOfElements();
  switch (example_dtype.value()) {
    case DataType::DE_INT64: {
      const auto data_buffer =
        reinterpret_cast<int64_t *>(column_tensor->GetMutableBuffer()) + tensor_index * num_elements;
      LimitedArraySlice<int64_t> slice(data_buffer, num_elements);
      if (!feature.ParseInt64List(&slice)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      if (slice.EndDistance() != 0) {
        return ReportUnexpectedDataShape(feature_name);
      }
      break;
    }
    case DataType::DE_FLOAT32: {
      const auto data_buffer =
        reinterpret_cast<float *>(column_tensor->GetMutableBuffer()) + tensor_index * num_elements;
      LimitedArraySlice<float> slice(data_buffer, num_elements);
      if (!feature.ParseFloatList(&slice)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      if (slice.EndDistance() != 0) {
        return ReportUnexpectedDataShape(feature_name);
      }
      break;
    }
    case DataType::DE_STRING: {
      const auto data_buffer = &(*string_col_map)[column_index][tensor_index * num_elements];
      LimitedArraySlice<std::string> slice(data_buffer, num_elements);
      if (!feature.ParseBytesList(&slice)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      if (slice.EndDistance() != 0) {
        return ReportUnexpectedDataShape(feature_name);
      }
      break;
    }
    default:
      return ReportUnexpectedDataType(feature_name, example_dtype);
  }
  return Status::OK();
}

Status PushStringToBuffer(const std::vector<std::string> &bytes_list, VarLenTensorBuffer *varlen_tensor_buffer,
                          const ColDescriptor &column_descriptor) {
  RETURN_UNEXPECTED_IF_NULL(varlen_tensor_buffer);
  if (column_descriptor.Type().value() == DataType::DE_STRING) {
    // check that each sample contains the same number of strings
    if (varlen_tensor_buffer->string_length != 0) {
      CHECK_FAIL_RETURN_UNEXPECTED(varlen_tensor_buffer->string_length == bytes_list.size(),
                                   "Could not batch string Tensors with different shapes.");
    } else {
      if (column_descriptor.Rank() != 0) {
        varlen_tensor_buffer->string_length = bytes_list.size();
      } else {
        varlen_tensor_buffer->string_length = 0;
      }
    }
    for (auto &bytes : bytes_list) {
      varlen_tensor_buffer->string_tensor.emplace_back(bytes);
    }
  } else if (column_descriptor.Type().value() == DataType::DE_UINT8) {
    size_t total_size = 0;
    for (auto &bytes : bytes_list) {
      total_size += bytes.size();
      varlen_tensor_buffer->string_tensor.emplace_back(bytes);
    }
    if (varlen_tensor_buffer->string_length != 0) {
      CHECK_FAIL_RETURN_UNEXPECTED(varlen_tensor_buffer->string_length == total_size,
                                   "Could not batch bytes Tensors with different shapes.");
    } else {
      varlen_tensor_buffer->string_length = total_size;
    }
  }
  return Status::OK();
}

Status ParseSerializedVarLenColumn(const parsed::Feature &feature, VarLenTensorBuffer *varlen_tensor_buffer,
                                   const StringPiece &feature_name, const ColDescriptor &column_descriptor,
                                   const DataType &example_dtype) {
  RETURN_UNEXPECTED_IF_NULL(varlen_tensor_buffer);
  bool type_cast_flag = false;
  if (example_dtype != column_descriptor.Type()) {
    const std::string msg =
      "The data type loaded from the example for feature name: " + column_descriptor.Name() +
      " does not match the predefined type in schema, the actual type: " + example_dtype.ToString() +
      ", but the predefined type: " + column_descriptor.Type().ToString();
    if (!example_dtype.IsString() && !column_descriptor.Type().IsString()) {
      MS_LOG(INFO) << msg << ". This will cause a type cast.";
      type_cast_flag = true;
    } else if (column_descriptor.Type().value() != DataType::DE_UINT8) {
      // allow to read data of type string or bytes into an uint8 tensor
      RETURN_STATUS_UNEXPECTED(msg);
    }
  }

  size_t num_elements;
  SmallVector<int64_t> int64_list;
  TensorVector<float> float_list;
  std::vector<std::string> bytes_list;
  switch (example_dtype.value()) {
    case DataType::DE_INT64: {
      if (!feature.ParseInt64List(&int64_list)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      num_elements = int64_list.size();
      break;
    }
    case DataType::DE_FLOAT32: {
      if (!feature.ParseFloatList(&float_list)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      num_elements = float_list.size();
      break;
    }
    case DataType::DE_STRING: {
      int actual_num_elements = 0;
      if (!feature.GetNumElementsInBytesList(&actual_num_elements)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      bytes_list.reserve(actual_num_elements);
      if (!feature.ParseBytesList(&bytes_list)) {
        return ReportUnexpectedParseFailure(feature_name);
      }
      num_elements = bytes_list.size();
      break;
    }
    default:
      return ReportUnexpectedDataType(feature_name, example_dtype);
  }

  TensorShape varlen_tensor_shape = TensorShape::CreateUnknownRankShape();
  RETURN_IF_NOT_OK(column_descriptor.MaterializeTensorShape(num_elements, &varlen_tensor_shape));
  std::shared_ptr<Tensor> varlen_tensor;
  switch (example_dtype.value()) {
    case DataType::DE_INT64: {
      RETURN_IF_NOT_OK(Tensor::CreateEmpty(varlen_tensor_shape, example_dtype, &varlen_tensor));
      CopyOrMoveBlock(int64_list.begin(), int64_list.end(),
                      reinterpret_cast<int64_t *>(varlen_tensor->GetMutableBuffer()));
      if (type_cast_flag) {
        std::shared_ptr<Tensor> casted_varlen_tensor;
        RETURN_IF_NOT_OK(TypeCast(varlen_tensor, &casted_varlen_tensor, column_descriptor.Type()));
        varlen_tensor_buffer->numeric_tensor.emplace_back(casted_varlen_tensor);
      } else {
        varlen_tensor_buffer->numeric_tensor.emplace_back(varlen_tensor);
      }
      break;
    }
    case DataType::DE_FLOAT32: {
      RETURN_IF_NOT_OK(Tensor::CreateFromTensor(std::shared_ptr<Tensor>(float_list.tensor()), &varlen_tensor));
      RETURN_IF_NOT_OK(varlen_tensor->Reshape(varlen_tensor_shape));
      if (type_cast_flag) {
        std::shared_ptr<Tensor> casted_varlen_tensor;
        RETURN_IF_NOT_OK(TypeCast(varlen_tensor, &casted_varlen_tensor, column_descriptor.Type()));
        varlen_tensor_buffer->numeric_tensor.emplace_back(casted_varlen_tensor);
      } else {
        varlen_tensor_buffer->numeric_tensor.emplace_back(varlen_tensor);
      }
      break;
    }
    case DataType::DE_STRING: {
      RETURN_IF_NOT_OK(PushStringToBuffer(bytes_list, varlen_tensor_buffer, column_descriptor));
      break;
    }
    default:
      return ReportUnexpectedDataType(feature_name, example_dtype);
  }
  return Status::OK();
}

Status ParseExampleOp::ParseSerializedExample(const std::string &example_bytes, TensorRow *parsed_row,
                                              std::unordered_map<int32_t, std::vector<std::string>> *string_column_map,
                                              std::vector<VarLenTensorBuffer> *varlen_tensor_vector,
                                              const size_t tensor_index) {
  RETURN_UNEXPECTED_IF_NULL(varlen_tensor_vector);
  parsed::Example parsed_example;
  CHECK_FAIL_RETURN_UNEXPECTED(ParseExample(example_bytes, &parsed_example),
                               "Failed to parse example bytes: " + example_bytes);

  const size_t parsed_example_size = parsed_example.size();
  std::vector<bool> feature_already_seen(data_schema_.NumColumns(), false);
  for (size_t i = 0; i < parsed_example_size; ++i) {
    // This is a logic that standard protobuf parsing is implementing.
    // I.e. last entry in the map overwrites all the previous ones.
    parsed::FeatureMapEntry &name_and_feature = parsed_example[parsed_example_size - i - 1];
    const StringPiece &feature_name = name_and_feature.first;
    parsed::Feature &feature = name_and_feature.second;

    if (column_name_id_map_.find(std::string(feature_name)) == column_name_id_map_.end()) {
      MS_LOG(INFO) << "Feature name: " << feature_name << " is not in schema, skip it.";
      continue;
    }

    DataType example_dtype;
    RETURN_IF_NOT_OK(feature.ParseDataType(&example_dtype));
    if (example_dtype == DataType::DE_UNKNOWN) {
      continue;
    }

    const auto column_index = column_name_id_map_[std::string(feature_name)];
    // If feature was already visited, skip.
    if (feature_already_seen[column_index]) {
      LogFeatureRepeated(feature_name);
      continue;
    }
    feature_already_seen[column_index] = true;

    const ColDescriptor &column_descriptor = data_schema_.Column(column_index);
    if (column_descriptor.HasKnownShape()) {
      RETURN_IF_NOT_OK(ParseSerializedKnownShapeColumn(feature, parsed_row, string_column_map, column_index,
                                                       tensor_index, feature_name, column_descriptor, example_dtype));
    } else {  // if variable length
      RETURN_IF_NOT_OK(ParseSerializedVarLenColumn(feature, &(*varlen_tensor_vector)[column_index], feature_name,
                                                   column_descriptor, example_dtype));
    }
  }

  for (int32_t column_index = 0; column_index < data_schema_.NumColumns(); ++column_index) {
    if (!feature_already_seen[column_index]) {
      RETURN_STATUS_UNEXPECTED("Feature name: " + data_schema_.Column(column_index).Name() +
                               " is required in schema but could not be found in tfrecord file.");
    }
  }
  return Status::OK();
}

Status ParseExampleOp::ConstructColumnMap(const std::string &example_bytes) {
  if (column_name_id_map_.empty()) {
    if (data_schema_.Empty()) {
      dataengine::Example example;
      if (!example.ParseFromString(example_bytes)) {
        RETURN_STATUS_UNEXPECTED("Failed to parse example bytes: " + std::string(example_bytes));
      }

      const dataengine::Features &example_features = example.features();
      const google::protobuf::Map<std::string, dataengine::Feature> &feature_map = example_features.feature();
      if (column_list_.empty()) {
        (void)std::transform(feature_map.begin(), feature_map.end(), std::back_inserter(column_list_),
                             [](const auto &it) -> std::string { return it.first; });
        std::sort(column_list_.begin(), column_list_.end());
      }

      for (const auto &column_name : column_list_) {
        auto it = feature_map.find(column_name);
        if (it == feature_map.end()) {
          RETURN_STATUS_UNEXPECTED("Invalid column list, failed to find column name: " + column_name + " in example.");
        }

        std::string column_type;
        const dataengine::Feature &feature = it->second;
        switch (feature.kind_case()) {
          case dataengine::Feature::KindCase::kBytesList:
            column_type = "uint8";
            break;
          case dataengine::Feature::KindCase::kFloatList:
            column_type = "float32";
            break;
          case dataengine::Feature::KindCase::kInt64List:
            column_type = "int64";
            break;
          default:
            RETURN_STATUS_UNEXPECTED("Unsupported column type, the column type of " + column_name +
                                     " should be int64, float32 or string.");
        }
        RETURN_IF_NOT_OK(
          data_schema_.AddColumn(ColDescriptor(column_name, DataType(column_type), TensorImpl::kFlexible, 1)));
      }
    }
    RETURN_IF_NOT_OK(data_schema_.GetColumnNameMap(&column_name_id_map_));
    CHECK_FAIL_RETURN_UNEXPECTED(!column_name_id_map_.empty(), "Can not get column name map, it is empty.");
  }
  return Status::OK();
}

void ParseExampleOp::CheckAndInitPool() {
  if (parallel_parse_ && pool_ == nullptr) {
    pool_ = std::make_unique<Eigen::ThreadPool>(kThreadPoolSize);
  }
}
}  // namespace mindspore::dataset
