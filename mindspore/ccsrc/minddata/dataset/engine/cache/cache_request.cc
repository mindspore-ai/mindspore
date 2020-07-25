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
#include "minddata/dataset/engine/cache/cache_request.h"

namespace mindspore {
namespace dataset {

Status CacheRowRequest::SerializeCacheRowRequest(const TensorRow &row) {
  buffers_.reserve(row.size() + 1);
  RETURN_IF_NOT_OK(SerializeTensorRowHeader(row));
  buffers_.push_back(fbb_->GetBufferPointer());
  for (const auto &ts : row) {
    buffers_.push_back(ts->GetBuffer());
  }
  return Status::OK();
}

Status CacheRowRequest::SerializeTensorRowHeader(const TensorRow &row) {
  try {
    fbb_ = std::make_shared<flatbuffers::FlatBufferBuilder>();
    std::vector<flatbuffers::Offset<TensorMetaMsg>> v;
    std::vector<int64_t> tensor_sz;
    v.reserve(row.size());
    tensor_sz.reserve(row.size());
    // We will go through each column in the row.
    for (const std::shared_ptr<Tensor> &ts_ptr : row) {
      flatbuffers::Offset<TensorMetaMsg> ts_off;
      RETURN_IF_NOT_OK(SerializeOneTensorMeta(ts_ptr, &ts_off));
      v.push_back(ts_off);
      tensor_sz.push_back(ts_ptr->SizeInBytes());
    }
    auto column_off = fbb_->CreateVector(v);
    auto data_sz_off = fbb_->CreateVector(tensor_sz);
    TensorRowHeaderMsgBuilder row_builder(*fbb_);
    row_builder.add_column(column_off);
    row_builder.add_data_sz(data_sz_off);
    // Pass the row_id even if it may not be known.
    row_builder.add_row_id(row.getId());
    row_builder.add_size_of_this(-1);  // fill in later after we call Finish.
    auto out = row_builder.Finish();
    fbb_->Finish(out);
    // Now go back to fill in size_of_this in the flat buffer.
    auto msg = GetMutableTensorRowHeaderMsg(fbb_->GetBufferPointer());
    auto success = msg->mutate_size_of_this(fbb_->GetSize());
    if (!success) {
      RETURN_STATUS_UNEXPECTED("Unable to set size_of_this");
    }
    return Status::OK();
  } catch (const std::bad_alloc &e) {
    return Status(StatusCode::kOutOfMemory, __LINE__, __FILE__);
  }
}

Status CacheRowRequest::SerializeOneTensorMeta(const std::shared_ptr<Tensor> &ts_ptr,
                                               flatbuffers::Offset<TensorMetaMsg> *out_off) {
  RETURN_UNEXPECTED_IF_NULL(out_off);
  const Tensor *ts = ts_ptr.get();
  auto shape_off = fbb_->CreateVector(ts->shape().AsVector());
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

  TensorMetaMsgBuilder ts_builder(*fbb_);
  ts_builder.add_dims(shape_off);
  ts_builder.add_type(dest);
  auto ts_off = ts_builder.Finish();
  *out_off = ts_off;
  return Status::OK();
}

Status BatchFetchRequest::RestoreOneTensor(const TensorMetaMsg *col_ts, const ReadableSlice &data,
                                           std::shared_ptr<Tensor> *out) {
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

Status BatchFetchRequest::RestoreRows(TensorTable *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  auto num_elements = row_id_.size();
  auto *offset_array = reinterpret_cast<const int64_t *>(mem_.GetPointer());
  TensorTable tbl;
  tbl.reserve(num_elements);
  ReadableSlice all(mem_.GetPointer(), mem_.GetSizeInBytes());
  for (auto i = 0; i < num_elements; ++i) {
    auto len = offset_array[i + 1] - offset_array[i];
    TensorRow row;
    row.setId(row_id_.at(i));
    if (len > 0) {
      ReadableSlice row_data(all, offset_array[i], len);
      // Next we de-serialize flat buffer to get back each column
      auto msg = GetTensorRowHeaderMsg(row_data.GetPointer());
      auto msg_sz = msg->size_of_this();
      // Start of the tensor data
      auto ts_offset = msg_sz;
      row.reserve(msg->column()->size());
      for (auto k = 0; k < msg->column()->size(); ++k) {
        auto col_ts = msg->column()->Get(k);
        std::shared_ptr<Tensor> ts;
        ReadableSlice data(row_data, ts_offset, msg->data_sz()->Get(k));
        RETURN_IF_NOT_OK(RestoreOneTensor(col_ts, data, &ts));
        row.push_back(ts);
        ts_offset += data.GetSize();
      }
    }
    tbl.push_back(std::move(row));
  }
  *out = std::move(tbl);
  return Status::OK();
}

Status CacheSchemaRequest::SerializeCacheSchemaRequest(const std::unordered_map<std::string, int32_t> &map) {
  try {
    fbb_ = std::make_shared<flatbuffers::FlatBufferBuilder>();
    std::vector<flatbuffers::Offset<ColumnNameMsg>> v;
    v.reserve(map.size());
    for (auto &column : map) {
      auto c = CreateColumnNameMsg(*fbb_, fbb_->CreateString(column.first), column.second);
      v.push_back(c);
    }
    auto v_off = fbb_->CreateVector(v);
    auto final_off = CreateSchemaMsg(*fbb_, v_off);
    fbb_->Finish(final_off);
    buf_ = fbb_->GetBufferPointer();
    len_of_buf_ = fbb_->GetSize();
    return Status::OK();
  } catch (const std::bad_alloc &e) {
    return Status(StatusCode::kOutOfMemory, __LINE__, __FILE__);
  }
}

std::unordered_map<std::string, int32_t> FetchSchemaRequest::GetColumnMap() {
  if (column_name_id_map_.empty()) {
    auto *map_msg = flatbuffers::GetRoot<SchemaMsg>(mem_.GetPointer());
    auto v = map_msg->column();
    for (auto i = 0; i < v->size(); ++i) {
      auto col = map_msg->column()->Get(i);
      column_name_id_map_.emplace(col->name()->str(), col->id());
    }
  }
  return column_name_id_map_;
}
}  // namespace dataset
}  // namespace mindspore
