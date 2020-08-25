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
#include "lite_cv/lite_mat.h"

namespace mindspore {
namespace dataset {

LiteMat::LiteMat() {
  data_ptr_ = 0;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = 0;
}

LiteMat::LiteMat(int width, LDataType data_type) {
  data_ptr_ = 0;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = 0;
  Init(width, data_type);
}

LiteMat::LiteMat(int width, int height, LDataType data_type) {
  data_ptr_ = 0;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = 0;
  Init(width, height, data_type);
}

LiteMat::LiteMat(int width, int height, int channel, LDataType data_type) {
  data_ptr_ = 0;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = 0;
  Init(width, height, channel, data_type);
}

LiteMat::~LiteMat() { Release(); }

int LiteMat::addRef(int *p, int value) {
  int v = *p;
  *p += value;
  return v;
}

LiteMat::LiteMat(const LiteMat &m) {
  data_ptr_ = m.data_ptr_;
  elem_size_ = m.elem_size_;
  width_ = m.width_;
  height_ = m.height_;
  channel_ = m.channel_;
  c_step_ = m.c_step_;
  dims_ = m.dims_;
  data_type_ = m.data_type_;
  ref_count_ = m.ref_count_;
  if (ref_count_) {
    addRef(ref_count_, 1);
  }
}

LiteMat &LiteMat::operator=(const LiteMat &m) {
  if (this == &m) {
    return *this;
  }

  if (m.ref_count_) {
    addRef(m.ref_count_, 1);
  }

  Release();
  data_ptr_ = m.data_ptr_;
  elem_size_ = m.elem_size_;
  width_ = m.width_;
  height_ = m.height_;
  channel_ = m.channel_;
  c_step_ = m.c_step_;
  dims_ = m.dims_;
  data_type_ = m.data_type_;
  ref_count_ = m.ref_count_;
  return *this;
}

void LiteMat::Init(int width, LDataType data_type) {
  Release();
  data_type_ = data_type;
  InitElemSize(data_type);
  width_ = width;
  dims_ = 1;
  height_ = 1;
  channel_ = 1;
  c_step_ = width;
  size_ = c_step_ * elem_size_;
  data_ptr_ = AlignMalloc(size_);
  ref_count_ = new int[1];
  *ref_count_ = 1;
}

void LiteMat::Init(int width, int height, LDataType data_type) {
  Release();
  data_type_ = data_type;
  InitElemSize(data_type);

  width_ = width;
  height_ = height;
  dims_ = 2;
  channel_ = 1;
  c_step_ = width_ * height_;
  size_ = c_step_ * elem_size_;
  data_ptr_ = AlignMalloc(size_);
  ref_count_ = new int[1];
  *ref_count_ = 1;
}

void LiteMat::Init(int width, int height, int channel, LDataType data_type) {
  Release();
  data_type_ = data_type;
  InitElemSize(data_type);
  width_ = width;
  height_ = height;
  dims_ = 3;
  channel_ = channel;
  c_step_ = ((height_ * width_ * elem_size_ + ALIGN - 1) & (-ALIGN)) / elem_size_;
  size_ = c_step_ * channel_ * elem_size_;

  data_ptr_ = AlignMalloc(size_);

  ref_count_ = new int[1];
  *ref_count_ = 1;
}

bool LiteMat::IsEmpty() const { return data_ptr_ == 0 || data_ptr_ == nullptr || c_step_ * channel_ == 0; }

void LiteMat::Release() {
  if (ref_count_ && (addRef(ref_count_, -1) == 1)) {
    if (data_ptr_) {
      AlignFree(data_ptr_);
    }
    if (ref_count_) {
      delete[] ref_count_;
    }
  }
  data_ptr_ = 0;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  ref_count_ = 0;
}

void *LiteMat::AlignMalloc(unsigned int size) {
  unsigned int length = sizeof(void *) + ALIGN - 1;
  void *p_raw = reinterpret_cast<void *>(malloc(size + length));
  if (p_raw) {
    void **p_algin = reinterpret_cast<void **>(((size_t)(p_raw) + length) & ~(ALIGN - 1));
    p_algin[-1] = p_raw;
    return p_algin;
  }
  return nullptr;
}
void LiteMat::AlignFree(void *ptr) { (void)free(reinterpret_cast<void **>(ptr)[-1]); }
inline void LiteMat::InitElemSize(LDataType data_type) {
  if (data_type == LDataType::UINT8) {
    elem_size_ = 1;
  } else if (data_type == LDataType::UINT16) {
    elem_size_ = 2;
  } else if (data_type == LDataType::FLOAT32) {
    elem_size_ = 4;
  } else {
  }
}
}  // namespace dataset
}  // namespace mindspore
