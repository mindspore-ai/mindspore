/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/engine/cache/cache_fbb.h"
namespace mindspore {
namespace dataset {
/// A private function used by SerializeTensorRowHeader to serialize each column in a tensor
/// \note Not to be called by outside world
/// \return Status object
Status SerializeOneTensorMeta(const std::shared_ptr<flatbuffers::FlatBufferBuilder> &fbb,
                              const std::shared_ptr<Tensor> &ts_ptr, flatbuffers::Offset<TensorMetaMsg> *out_off) {
  RETURN_UNEXPECTED_IF_NULL(out_off);
  const Tensor *ts = ts_ptr.get();
  auto shape_off = fbb->CreateVector(ts->shape().AsVector());
  const auto ptr = ts->GetBuffer();
  if (ptr == nullptr) {
    RETURN_STATUS_UNEXPECTED("Tensor buffer is null");
  }
  auto src = ts->type().value();
  TensorType dest;
#define CASE(t)                        \
  case DataType::t:                    \
    dest = TensorType::TensorType_##t; \
    break
  // Map the type to fill in the flat buffer.
  switch (src) {
    CASE(DE_BOOL);
    CASE(DE_INT8);
    CASE(DE_UINT8);
    CASE(DE_INT16);
    CASE(DE_UINT16);
    CASE(DE_INT32);
    CASE(DE_UINT32);
    CASE(DE_INT64);
    CASE(DE_UINT64);
    CASE(DE_FLOAT16);
    CASE(DE_FLOAT32);
    CASE(DE_FLOAT64);
    CASE(DE_STRING);
    default:
      MS_LOG(ERROR) << "Unknown tensor. Dumping content:\n" << *ts;
      RETURN_STATUS_UNEXPECTED("Unknown type");
  }
#undef CASE

  TensorMetaMsgBuilder ts_builder(*fbb);
  ts_builder.add_dims(shape_off);
  ts_builder.add_type(dest);
  auto ts_off = ts_builder.Finish();
  *out_off = ts_off;
  return Status::OK();
}

Status SerializeTensorRowHeader(const TensorRow &row, std::shared_ptr<flatbuffers::FlatBufferBuilder> *out_fbb) {
  RETURN_UNEXPECTED_IF_NULL(out_fbb);
  auto fbb = std::make_shared<flatbuffers::FlatBufferBuilder>();
  try {
    fbb = std::make_shared<flatbuffers::FlatBufferBuilder>();
    std::vector<flatbuffers::Offset<TensorMetaMsg>> v;
    std::vector<int64_t> tensor_sz;
    v.reserve(row.size());
    tensor_sz.reserve(row.size());
    // We will go through each column in the row.
    for (const std::shared_ptr<Tensor> &ts_ptr : row) {
      flatbuffers::Offset<TensorMetaMsg> ts_off;
      RETURN_IF_NOT_OK(SerializeOneTensorMeta(fbb, ts_ptr, &ts_off));
      v.push_back(ts_off);
      tensor_sz.push_back(ts_ptr->SizeInBytes());
    }
    auto column_off = fbb->CreateVector(v);
    auto data_sz_off = fbb->CreateVector(tensor_sz);
    TensorRowHeaderMsgBuilder row_builder(*fbb);
    row_builder.add_column(column_off);
    row_builder.add_data_sz(data_sz_off);
    // Pass the row_id even if it may not be known.
    row_builder.add_row_id(row.getId());
    row_builder.add_size_of_this(-1);  // fill in later after we call Finish.
    auto out = row_builder.Finish();
    fbb->Finish(out);
    // Now go back to fill in size_of_this in the flat buffer.
    auto msg = GetMutableTensorRowHeaderMsg(fbb->GetBufferPointer());
    auto success = msg->mutate_size_of_this(fbb->GetSize());
    if (!success) {
      RETURN_STATUS_UNEXPECTED("Unable to set size_of_this");
    }
    (*out_fbb) = std::move(fbb);
    return Status::OK();
  } catch (const std::bad_alloc &e) {
    return Status(StatusCode::kMDOutOfMemory, __LINE__, __FILE__);
  }
}

Status RestoreOneTensor(const TensorMetaMsg *col_ts, const ReadableSlice &data, std::shared_ptr<Tensor> *out) {
  RETURN_UNEXPECTED_IF_NULL(col_ts);
  auto shape_in = col_ts->dims();
  auto type_in = col_ts->type();
  std::vector<dsize_t> v;
  v.reserve(shape_in->size());
  v.assign(shape_in->begin(), shape_in->end());
  TensorShape shape(v);
  DataType::Type dest = DataType::DE_UNKNOWN;
#define CASE(t)               \
  case TensorType_##t:        \
    dest = DataType::Type::t; \
    break

  switch (type_in) {
    CASE(DE_BOOL);
    CASE(DE_INT8);
    CASE(DE_UINT8);
    CASE(DE_INT16);
    CASE(DE_UINT16);
    CASE(DE_INT32);
    CASE(DE_UINT32);
    CASE(DE_INT64);
    CASE(DE_UINT64);
    CASE(DE_FLOAT16);
    CASE(DE_FLOAT32);
    CASE(DE_FLOAT64);
    CASE(DE_STRING);
  }
#undef CASE

  DataType type(dest);
  std::shared_ptr<Tensor> ts;
  RETURN_IF_NOT_OK(
    Tensor::CreateFromMemory(shape, type, static_cast<const unsigned char *>(data.GetPointer()), data.GetSize(), &ts));
  // Next we restore the real data which can be embedded or stored separately.
  if (ts->SizeInBytes() != data.GetSize()) {
    MS_LOG(ERROR) << "Unexpected length. Read " << data.GetSize() << ". Expected " << ts->SizeInBytes() << ".\n"
                  << "Dumping tensor\n"
                  << *ts << "\n";
    RETURN_STATUS_UNEXPECTED("Length mismatch. See log file for details.");
  }
  *out = std::move(ts);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
