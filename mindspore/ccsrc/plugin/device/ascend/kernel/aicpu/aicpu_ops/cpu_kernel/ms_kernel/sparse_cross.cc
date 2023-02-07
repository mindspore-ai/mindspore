/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "cpu_ops_kernel.h"
#include <string>
#include "sparse_cross.h"
#include <iostream>

namespace {
static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;
const char *kSparseCross = "SparseCross";
}  // namespace

namespace aicpu {
typedef std::pair<uint64_t, uint64_t> uint128_t;
inline uint64_t Uint128Low64(const uint128_t x) { return x.first; }
inline uint64_t Uint128High64(const uint128_t x) { return x.second; }
inline uint128_t Uint128(uint64_t lo, uint64_t hi) { return uint128_t(lo, hi); }
#define STATIC_INLINE static inline

using namespace std;

using ui = unsigned int;
using ul = unsigned long;
using uc = unsigned char;
using ull = unsigned long long;

static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;

STATIC_INLINE uint64_t Fetch64(const char *p) {
  uint64_t result;
  memcpy(&result, p, sizeof(result));
  return uint64_in_expected_order(result);
}

STATIC_INLINE uint32_t Fetch32(const char *p) {
  uint32_t result;
  memcpy(&result, p, sizeof(result));
  return uint32_in_expected_order(result);
}

STATIC_INLINE uint64_t Hash128to64(uint128_t x) {
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (Uint128Low64(x) ^ Uint128High64(x)) * kMul;
  uint64_t value = 47;
  a ^= (a >> value);
  uint64_t b = (Uint128High64(x) ^ a) * kMul;
  b ^= (b >> value);
  b *= kMul;
  return b;
}

STATIC_INLINE uint64_t ShiftMix(uint64_t val) {
  uint64_t value = 47;
  return val ^ (val >> value);
}

STATIC_INLINE uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t mul) {
  uint64_t a = (u ^ v) * mul;
  uint64_t value = 47;
  a ^= (a >> value);
  uint64_t b = (v ^ a) * mul;
  b ^= (b >> value);
  b *= mul;
  return b;
}

STATIC_INLINE uint64_t HashLen0to16(const char *s, size_t len) {
  if (len > 0) {
    uint8_t a = s[0];
    uint8_t b = s[len >> 1];
    uint8_t c = s[len - 1];
    uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
    uint32_t z = len + (static_cast<uint32_t>(c) << 2);
    return ShiftMix(y * k2 ^ z * k0) * k2;
  }
  return k2;
}

uint64_t FarmHash64(const char *s, size_t len) { return HashLen0to16(s, len); }

uint64_t Fingerprint64(const string s) { return FarmHash64(s.data(), s.size()); }

template <typename InternalType>
class ColumnInterface {
 public:
  virtual int64_t FeatureCount(int64_t batch) const = 0;
  virtual InternalType Feature(int64_t batch, int64_t n) const = 0;
  virtual ~ColumnInterface() {}
};

template <typename InternalType>
class SparseTensorColumn : public ColumnInterface<InternalType> {
 public:
  SparseTensorColumn(Tensor *values, std::vector<int64_t> feature_counts, std::vector<int64_t> feature_start_indices)
      : values_(values),
        feature_counts_(std::move(feature_counts)),
        feature_start_indices_(std::move(feature_start_indices)) {
    if (feature_counts_.size() != feature_start_indices_.size()) {
      KERNEL_LOG_ERROR("feature_counts_ is not equal to feature_start_indices_.");
    }
  }
  int64_t FeatureCount(int64_t batch) const override { return feature_counts_[batch]; }
  InternalType Feature(int64_t batch, int64_t n) const override;
  ~SparseTensorColumn() override {}

 private:
  Tensor *values_;
  std::vector<int64_t> feature_counts_;
  std::vector<int64_t> feature_start_indices_;
};

template <>
std::string SparseTensorColumn<std::string>::Feature(int64_t batch, int64_t n) const {
  const int64_t start = feature_start_indices_[batch];
  EigenTensor values_e(values_, values_->GetData());
  if (DT_STRING == values_->GetDataType()) return values_e.vec<std::string>().data()[start + n];
  return std::to_string(values_e.vec<int64_t>().data()[start + n]);
}

template <>
int64_t SparseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n) const {
  const int64_t start = feature_start_indices_[batch];
  EigenTensor values_e(values_, values_->GetData());
  if (DT_STRING == values_->GetDataType()) {
    return Fingerprint64(values_e.vec<std::string>().data()[start + n]);
  }
  return values_e.vec<int64_t>().data()[start + n];
}

template <typename InternalType>
class DenseTensorColumn : public ColumnInterface<InternalType> {
 public:
  explicit DenseTensorColumn(Tensor *tensor) : tensor_(tensor) {}
  int64_t FeatureCount(int64_t batch) const override { return tensor_->GetTensorShape()->GetDimSize(1); }
  InternalType Feature(int64_t batch, int64_t n) const override;
  ~DenseTensorColumn() override {}

 private:
  Tensor *tensor_;
};

template <>
int64_t DenseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n) const {
  EigenTensor tensor_e(tensor_, tensor_->GetData());
  if (DT_STRING == tensor_->GetDataType()) return Fingerprint64(tensor_e.matrix<std::string>()(batch, n));
  return tensor_e.matrix<int64_t>()(batch, n);
}

template <>
std::string DenseTensorColumn<std::string>::Feature(int64_t batch, int64_t n) const {
  EigenTensor tensor_e(tensor_, tensor_->GetData());
  if (DT_STRING == tensor_->GetDataType()) return tensor_e.matrix<std::string>()(batch, n);
  return std::to_string(tensor_e.matrix<int64_t>()(batch, n));
}

template <typename OutType>
class OutputUpdater {
 public:
  OutputUpdater(const std::vector<int64_t> &output_start_indices, Tensor *indices_out, Tensor *values_out)
      : output_start_indices_(output_start_indices), indices_out_(indices_out), values_out_(values_out) {}
  void Update(const int64_t batch_index, const int64_t cross_count, const OutType &cross) const {
    const int64_t output_index = output_start_indices_[batch_index] + cross_count;
    auto indices_out_addr = static_cast<int64_t *>(indices_out_->GetData());
    int64_t value = 2;
    indices_out_addr[output_index * value] = batch_index;
    indices_out_addr[output_index * value + 1] = cross_count;
    auto values_out_addr = static_cast<OutType *>(values_out_->GetData());
    values_out_addr[output_index] = cross;
  }

 private:
  const std::vector<int64_t> &output_start_indices_;
  Tensor *indices_out_;
  Tensor *values_out_;
};

template <typename InternalType>
class StringCrosser {
 public:
  StringCrosser(const std::vector<std::unique_ptr<ColumnInterface<InternalType>>> &columns,
                const int64_t num_buckets_unused, const uint64_t hash_key_unused)
      : columns_(columns) {}
  std::string Generate(const int64_t batch_index, const std::vector<int64_t> &permutation) const {
    static const auto k_feature_separator = "_X_";
    std::vector<InternalType> cross_vec(columns_.size());
    for (size_t i = 0; i < permutation.size(); i++) {
      cross_vec[i] = columns_[i]->Feature(batch_index, permutation[i]);
    }
    size_t i;
    string str1 = "";
    for (i = 0; i < cross_vec.size() - 1; i++) {
      str1 = str1 + cross_vec[i].data();
      str1 = str1 + k_feature_separator;
    }
    str1 = str1 + cross_vec[i].data();
    return str1;
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<InternalType>>> &columns_;
};

class HashCrosser {
 public:
  HashCrosser(const std::vector<std::unique_ptr<ColumnInterface<int64_t>>> &columns, const int64_t num_buckets,
              const uint64_t hash_key)
      : columns_(columns), num_buckets_(num_buckets), hash_key_(hash_key) {}

  uint64_t ShiftMix(const uint64_t val) const { return val ^ (val >> 47); }
  uint64_t FingerprintCat64(const uint64_t fp1, const uint64_t fp2) const {
    static const uint64_t kMul = 0xc6a4a7935bd1e995ULL;
    uint64_t result = fp1 ^ kMul;
    result ^= ShiftMix(fp2 * kMul) * kMul;
    result *= kMul;
    result = ShiftMix(result) * kMul;
    result = ShiftMix(result);
    return result;
  }

  int64_t Generate(const int64_t batch_index, const std::vector<int64_t> &permutation) const {
    uint64_t hashed_output = hash_key_;
    for (size_t i = 0; i < permutation.size(); ++i) {
      uint64_t hash_i = columns_[i]->Feature(batch_index, permutation[i]);
      hashed_output = FingerprintCat64(hashed_output, hash_i);
    }
    if (num_buckets_ > 0) {
      return hashed_output % num_buckets_;
    } else {
      return hashed_output % std::numeric_limits<int64_t>::max();
    }
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<int64_t>>> &columns_;
  const int64_t num_buckets_;
  const uint64_t hash_key_;
};

template <typename InternalType>
class ProductIterator {
 public:
  explicit ProductIterator(const std::vector<std::unique_ptr<ColumnInterface<InternalType>>> &columns,
                           int64_t batch_index)
      : columns_(columns), batch_index_(batch_index) {
    next_permutation_.resize(columns_.size(), 0);
    has_next_ = true;
    for (size_t i = 0; i < columns_.size(); i++) {
      if (columns_[i]->FeatureCount(batch_index_) == 0) {
        has_next_ = false;
        break;
      }
    }
  }
  std::vector<int64_t> Next() {
    std::vector<int64_t> permutation(next_permutation_);
    bool carry = true;
    for (int64_t i = next_permutation_.size() - 1; i >= 0; i--) {
      if (carry) {
        next_permutation_[i] = next_permutation_[i] + 1;
      }
      if (next_permutation_[i] == columns_[i]->FeatureCount(batch_index_)) {
        next_permutation_[i] = 0;
      } else {
        carry = false;
        break;
      }
    }
    has_next_ = !carry;
    return permutation;
  }
  bool HasNext() { return has_next_; }

 private:
  bool has_next_;
  const std::vector<std::unique_ptr<ColumnInterface<InternalType>>> &columns_;
  const int64_t batch_index_;
  std::vector<int64_t> next_permutation_;
};

template <bool HASHED_OUTPUT, typename InternalType>
struct CrossTraits;
template <typename InternalType>
struct CrossTraits<false, InternalType> {
  typedef StringCrosser<InternalType> Crosser;
  typedef OutputUpdater<std::string> Updater;
};

template <>
struct CrossTraits<true, int64_t> {
  typedef HashCrosser Crosser;
  typedef OutputUpdater<int64_t> Updater;
};

int64_t CalculateBatchSize(const OpInputList &shapes_list_in, const OpInputList &dense_list_in) {
  EigenTensor shapes_list_in_e(shapes_list_in[0], shapes_list_in[0]->GetData());
  if (shapes_list_in.size() > 0) {
    return shapes_list_in_e.vec<int64_t>()(0);
  }
  if (dense_list_in.size() > 0) {
    return dense_list_in[0]->GetTensorShape()->GetDimSize(0);
  }
  return 0;
}

void ExtractFeatureData(const OpInputList &indices_list_in, int64_t batch_size,
                        std::vector<std::vector<int64_t>> *feature_counts,
                        std::vector<std::vector<int64_t>> *feature_start_indices) {
  std::vector<int64_t> current_row(indices_list_in.size());
  for (int64_t b = 0; b < batch_size; b++) {
    for (int64_t i = 0; i < indices_list_in.size(); i++) {
      EigenTensor indices_list_in_e(indices_list_in[i], indices_list_in[i]->GetData());
      const auto indices = indices_list_in_e.matrix<int64_t>();
      int64_t feature_count = 0;
      int64_t start_index = current_row[i];
      while (current_row[i] < indices_list_in[i]->GetTensorShape()->GetDimSize(0) && indices(current_row[i], 0) == b) {
        feature_count++;
        current_row[i]++;
      }
      (*feature_counts)[i].push_back(feature_count);
      (*feature_start_indices)[i].push_back(start_index);
    }
  }
}

template <typename InternalType>
std::vector<std::unique_ptr<ColumnInterface<InternalType>>> ColumnsFromInput(const OpInputList &indices_list_in,
                                                                             const OpInputList &values_list_in,
                                                                             const OpInputList &shapes_list_in,
                                                                             const OpInputList &dense_list_in) {
  std::vector<std::unique_ptr<ColumnInterface<InternalType>>> columns;
  const int64_t batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
  const int64_t number_of_columns = shapes_list_in.size();
  std::vector<std::vector<int64_t>> feature_counts(number_of_columns, std::vector<int64_t>());
  std::vector<std::vector<int64_t>> feature_start_indices(number_of_columns, std::vector<int64_t>());
  ExtractFeatureData(indices_list_in, batch_size, &feature_counts, &feature_start_indices);
  columns.reserve(values_list_in.size());
  for (int64_t i = 0; i < values_list_in.size(); ++i) {
    columns.emplace_back(
      new SparseTensorColumn<InternalType>(values_list_in[i], feature_counts[i], feature_start_indices[i]));
  }
  for (int64_t i = 0; i < dense_list_in.size(); ++i) {
    columns.emplace_back(new DenseTensorColumn<InternalType>(dense_list_in[i]));
  }
  return columns;
}

template <typename InternalType>
int64_t CrossCountByBatchIndex(const std::vector<std::unique_ptr<ColumnInterface<InternalType>>> &columns,
                               int64_t batch_index) {
  int64_t cross_count = 1;
  for (size_t i = 0; i < columns.size(); i++) {
    const auto feature_count = columns[i]->FeatureCount(batch_index);
    if (feature_count == 0) {
      return 0;
    }
    cross_count *= feature_count;
  }
  return cross_count;
}

template <typename InternalType>
void CreateOutputTensors(const std::vector<std::unique_ptr<ColumnInterface<InternalType>>> &columns, int64_t batch_size,
                         CpuKernelContext *context, Tensor *indices_out, Tensor *values_out, Tensor *shape_out,
                         std::vector<int64_t> *output_start_indices) {
  int64_t cross_count_total = 0;
  int64_t max_cross_count = 0;
  for (int64_t b = 0; b < batch_size; b++) {
    (*output_start_indices)[b] = cross_count_total;
    const auto cross_count = CrossCountByBatchIndex<InternalType>(columns, b);
    max_cross_count = std::max(max_cross_count, cross_count);
    cross_count_total += cross_count;
  }
  indices_out = context->Output(0);
  std::vector<int64_t> indices_t;
  int64_t value1 = 1;
  int64_t value2 = 2;
  indices_t.reserve(value2);
  indices_t.push_back(cross_count_total);
  indices_t.push_back(value2);
  indices_out->GetTensorShape()->SetDimSizes(indices_t);
  indices_out->SetDataType(DT_INT64);

  values_out = context->Output(value1);
  std::vector<int64_t> values_t;
  values_t.reserve(value1);
  values_t.push_back(cross_count_total);
  values_out->GetTensorShape()->SetDimSizes(values_t);

  shape_out = context->Output(value2);
  std::vector<int64_t> shape_t;
  shape_t.reserve(value1);
  shape_t.push_back(value2);
  auto shape_vec = static_cast<int64_t *>(shape_out->GetData());
  shape_vec[0] = batch_size;
  shape_vec[1] = max_cross_count;
  shape_out->GetTensorShape()->SetDimSizes(shape_t);
}

template <bool HASHED_OUTPUT, typename InternalType>
uint32_t SparseCrossCpuKernel::SparseCrossCompute(CpuKernelContext &ctx) {
  auto num_buckets_ptr = ctx.GetAttr("num_buckets");
  uint32_t inputSize = ctx.GetInputsSize();
  int64_t num_buckets_ = 0;
  int64_t num = inputSize / 3;
  uint64_t hash_key_ = ctx.GetAttr("hash_key")->GetInt();
  auto num_ptr = ctx.GetAttr("N");
  if (num_ptr != nullptr) {
    num = num_ptr->GetInt();
  } else {
    if (inputSize % 3 == 0) num = num - 1;
  }
  if (num_buckets_ptr != nullptr) {
    num_buckets_ = num_buckets_ptr->GetInt();
  }
  uint32_t start1 = 0;
  uint32_t stop = num;
  OpInputList indices_list_in(&ctx, start1, stop);
  start1 = start1 + num;
  stop = start1 + num;
  OpInputList values_list_in(&ctx, start1, stop);
  start1 = start1 + num;
  stop = start1 + num;
  OpInputList shapes_list_in(&ctx, start1, stop);
  start1 = start1 + num;
  OpInputList dense_list_in(&ctx, start1, inputSize);
  const auto size = indices_list_in.size();
  int64_t value = 2;
  for (int64_t i = 0; i < size; i++) {
    if (indices_list_in[i]->GetTensorShape()->GetDimSize(1) != value) {
      KERNEL_LOG_ERROR("Expected D2 of index to be 2 got [%d], at position [%d].",
                       indices_list_in[i]->GetTensorShape()->GetDimSize(1), i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  for (int64_t i = 0; i < size; i++) {
    if (indices_list_in[i]->GetTensorShape()->GetDimSize(0) != values_list_in[i]->GetTensorShape()->GetDimSize(0)) {
      KERNEL_LOG_ERROR("Expected size of values to be [%d], but got [%d] at position [%d].",
                       indices_list_in[i]->GetTensorShape()->GetDimSize(0),
                       values_list_in[i]->GetTensorShape()->GetDimSize(0), i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  const auto batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
  for (int64_t i = 0; i < size; i++) {
    EigenTensor shapes_list_in_e(shapes_list_in[i], shapes_list_in[i]->GetData());
    int64_t value = 2;
    if (shapes_list_in_e.vec<int64_t>().size() != value) {
      KERNEL_LOG_ERROR("shape should imply a 2D tensor, but got [%d].", shapes_list_in[i]->GetTensorShape());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  for (int64_t i = 0; i < dense_list_in.size(); ++i) {
    if (dense_list_in[i]->GetTensorShape()->GetDimSize(0) != batch_size) {
      KERNEL_LOG_ERROR("Expected batch size [%d],got [%d].", batch_size,
                       dense_list_in[i]->GetTensorShape()->GetDimSize(0));
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  std::vector<std::unique_ptr<ColumnInterface<InternalType>>> columns =
    ColumnsFromInput<InternalType>(indices_list_in, values_list_in, shapes_list_in, dense_list_in);
  typename CrossTraits<HASHED_OUTPUT, InternalType>::Crosser crosser(columns, num_buckets_, hash_key_);
  Tensor *indices_out = ctx.Output(0);
  Tensor *values_out = ctx.Output(1);
  Tensor *shape_out = ctx.Output(2);
  std::vector<int64_t> output_start_indices(batch_size);
  CreateOutputTensors(columns, batch_size, &ctx, indices_out, values_out, shape_out, &output_start_indices);
  typename CrossTraits<HASHED_OUTPUT, InternalType>::Updater updater(output_start_indices, indices_out, values_out);
  for (int64_t b = 0; b < batch_size; b++) {
    ProductIterator<InternalType> product_iterator(columns, b);
    int64_t cross_count = 0;
    while (product_iterator.HasNext()) {
      const auto permutation = product_iterator.Next();

      updater.Update(b, cross_count, crosser.Generate(b, permutation));
      cross_count++;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t SparseCrossCpuKernel::Compute(CpuKernelContext &ctx) {
  bool hash_out = ctx.GetAttr("hashed_output")->GetBool();
  DataType intertype = ctx.GetAttr("internal_type")->GetDataType();
  if (hash_out == 0) {
    if (intertype == 0) {
      uint32_t res = SparseCrossCompute<false, string>(ctx);
      if (res == 1) {
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  } else if (hash_out == 1) {
    uint32_t res = SparseCrossCompute<true, int64_t>(ctx);
    if (res == 1) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseCross, SparseCrossCpuKernel);
}  // namespace aicpu