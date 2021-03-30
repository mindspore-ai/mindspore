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

#include "src/runtime/gpu/opencl/opencl_allocator.h"
#include <utility>
#include "src/runtime/gpu/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore::lite::opencl {

OpenCLAllocator::OpenCLAllocator(OpenCLRuntime *ocl_runtime) : ocl_runtime_(ocl_runtime) {}

OpenCLAllocator::~OpenCLAllocator() { Clear(); }

void OpenCLAllocator::SetContext(const AllocatorContext &ctx) {
  lock_flag_ = ctx.lockFlag;
  shift_factor_ = ctx.shiftFactor;
}

void OpenCLAllocator::Lock() {
  if (lock_flag_) {
    lock.lock();
  }
}

void OpenCLAllocator::UnLock() {
  if (lock_flag_) {
    lock.unlock();
  }
}

void *OpenCLAllocator::MinimumFit(MemType mem_type, size_t size, const ImageSize &img_size) {
  auto iter = free_list_.lower_bound(size);
  while (iter != free_list_.end() && (iter->second->size_ >= size) && (iter->second->size_ < (size << shift_factor_))) {
    auto mem_buf = iter->second;
    bool is_match = mem_buf->mem_type_ == mem_type;
    if (mem_type == MemType::IMG) {
      is_match &= mem_buf->device_ptr_ != nullptr;
      is_match &= mem_buf->img_size_ == img_size;
    }
    if (is_match) {
      free_list_.erase(iter);
      allocated_list_[mem_buf->host_ptr_] = mem_buf;
      MS_LOG(DEBUG) << "Find Mem from free list. size: " << mem_buf->size_ << ", host addr: " << mem_buf->host_ptr_
                    << ", device addr: " << mem_buf->device_ptr_;
      return mem_buf->host_ptr_;
    }
    ++iter;
  }
  return nullptr;
}

void *OpenCLAllocator::CreateBuffer(size_t size, void *data, size_t flags, cl::Buffer **buffer) {
  cl_int ret = CL_SUCCESS;
  MS_ASSERT(buffer);
  MS_ASSERT(size > 0);
  *buffer = new (std::nothrow) cl::Buffer(*ocl_runtime_->Context(), static_cast<cl_mem_flags>(flags), size, data, &ret);
  if (*buffer == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL buffer failed! (ERROR CODE: " << ret << ")";
    return nullptr;
  }
  void *host_ptr = ocl_runtime_->MapBuffer(**buffer, CL_MAP_READ | CL_MAP_WRITE, size);
  MS_ASSERT(host_ptr);
  if (host_ptr == nullptr) {
    delete *buffer;
    MS_LOG(ERROR) << "Map buffer failed, can not found buffer :" << *buffer << ", host_ptr=" << host_ptr;
    return nullptr;
  }
  cl::Memory *mem = *buffer;
  MS_ASSERT(mem);
  ret = ocl_runtime_->UnmapBuffer(*mem, host_ptr);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "UnmapBuffer failed.";
  }
  return host_ptr;
}

void *OpenCLAllocator::CreateImage2D(size_t size, const ImageSize &img_size, void *data, size_t flags, bool is_map,
                                     cl::Buffer **buffer, cl::Image2D **image) {
  cl_int ret = CL_SUCCESS;
  MS_ASSERT(buffer);
  MS_ASSERT(image);
  if (data == nullptr) {
    // copy from cl2.hpp
    cl_image_desc desc = {CL_MEM_OBJECT_IMAGE2D, img_size.width, img_size.height, 0, 0, 0, 0, 0, 0, (**buffer).get()};
    const cl::Context &context = *ocl_runtime_->Context();
    cl_image_format image_format{CL_RGBA, static_cast<uint32_t>(img_size.dtype)};
    *image = new (std::nothrow) cl::Image2D(clCreateImage(context.get(), 0, &image_format, &desc, nullptr, &ret));
  } else {
    cl::ImageFormat image_format(CL_RGBA, img_size.dtype);
    *image = new (std::nothrow) cl::Image2D(*ocl_runtime_->Context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            image_format, img_size.width, img_size.height, 0, data, &ret);
  }
  if (*image == nullptr) {
    delete *buffer;
    MS_LOG(ERROR) << "Create OpenCL Image2D failed! (ERROR CODE: " << mindspore::kernel::CLErrorCode(ret) << ")";
    return nullptr;
  }
  if (ret != CL_SUCCESS) {
    delete *buffer;
    delete *image;
    MS_LOG(ERROR) << "Create OpenCL Image2D  (ERROR CODE: " << mindspore::kernel::CLErrorCode(ret) << ")";
    return nullptr;
  }
  MS_LOG(DEBUG) << "Malloc a new Image2D, width=" << img_size.width << ", height=" << img_size.height;
  void *host_ptr = nullptr;
  if (is_map) {
    std::vector<size_t> region{img_size.width, img_size.height, 1};
    host_ptr = ocl_runtime_->MapBuffer(**image, true, CL_MAP_READ | CL_MAP_WRITE, region);
    if (host_ptr == nullptr) {
      delete *buffer;
      delete *image;
      MS_LOG(ERROR) << "Map image failed, can not found image :" << *image << ", host_ptr=" << host_ptr;
      return nullptr;
    }
    cl::Memory *mem = *image;
    ret = ocl_runtime_->UnmapBuffer(*mem, host_ptr);
    if (ret != CL_SUCCESS) {
      MS_LOG(WARNING) << "UnmapBuffer failed.";
    }
  }
  return host_ptr;
}

int OpenCLAllocator::GetImgDtypeSize(const ImageSize &img_size) {
  size_t dtype_size = 0;
  if (img_size.dtype == CL_FLOAT) {
    dtype_size = sizeof(cl_float);
  } else if (img_size.dtype == CL_HALF_FLOAT) {
    dtype_size = sizeof(cl_half);
  } else if (img_size.dtype == CL_UNSIGNED_INT8) {
    dtype_size = sizeof(cl_uchar);
  } else {
    MS_LOG(ERROR) << "Unsupported dtype " << img_size.dtype;
    return RET_ERROR;
  }
  uint32_t image_alignment = ocl_runtime_->GetImagePitchAlignment();
  size_t size = UP_ROUND(img_size.width, image_alignment) * img_size.height * C4NUM * dtype_size;
  return size;
}

void *OpenCLAllocator::_Malloc(MemType mem_type, void *data, size_t size, const ImageSize &img_size) {
  auto svm_capabilities = ocl_runtime_->GetSVMCapabilities();
  if (mem_type == MemType::IMG) {
    size = GetImgDtypeSize(img_size);
  }
  if (size > ocl_runtime_->GetMaxAllocSize()) {
    MS_LOG(ERROR) << "MallocData out of max_size, size: " << size;
    return nullptr;
  }
  Lock();
  void *host_ptr = MinimumFit(mem_type, size, img_size);
  if (host_ptr != nullptr && data == nullptr) {
    UnLock();
    return host_ptr;
  }
  total_size_ += size;
  const uint64_t max_size = ocl_runtime_->GetGlobalMemSize() * 0.8;
  if (total_size_ >= max_size) {
    UnLock();
    MS_LOG(ERROR) << "Mem pool out of max_size, total size: " << total_size_ << ", max size: " << max_size;
    return nullptr;
  }
  cl::Buffer *buffer = nullptr;
  cl::Image2D *image = nullptr;
  cl_mem_flags flags = CL_MEM_READ_WRITE;
  if (svm_capabilities) {
    flags |= (svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0;
    flags |= (svm_capabilities & CL_DEVICE_SVM_ATOMICS) ? CL_MEM_SVM_ATOMICS : 0;
    host_ptr = clSVMAlloc((*ocl_runtime_->Context())(), flags, size, 0);
  } else {
    flags |= (data == nullptr) ? CL_MEM_ALLOC_HOST_PTR : CL_MEM_COPY_HOST_PTR;
    if (mem_type == MemType::BUF || data == nullptr) {
      host_ptr = CreateBuffer(size, data, flags, &buffer);
      if (host_ptr == nullptr) {
        UnLock();
        return nullptr;
      }
    }
    if (mem_type == MemType::IMG) {
      void *host_ptr_im = CreateImage2D(size, img_size, data, flags, data != nullptr, &buffer, &image);
      if (data != nullptr && host_ptr_im == nullptr) {
        UnLock();
        return nullptr;
      }
      host_ptr = (data != nullptr) ? host_ptr_im : host_ptr;
    }
  }
  MemBuf *mem_buf = new (std::nothrow) MemBuf;
  if (mem_buf == nullptr) {
    delete buffer;
    delete image;
    UnLock();
    return nullptr;
  }
  mem_buf->size_ = size;
  mem_buf->device_ptr_ = static_cast<void *>(buffer);
  mem_buf->host_ptr_ = host_ptr;
  mem_buf->image_ptr_ = static_cast<void *>(image);
  mem_buf->mem_type_ = mem_type;
  mem_buf->img_size_ = img_size;
  allocated_list_[host_ptr] = mem_buf;
  UnLock();
  std::string type_name = mem_type == MemType::BUF ? "buffer" : "Image2D";
  MS_LOG(DEBUG) << "Malloc a new " << type_name << ". size: " << mem_buf->size_ << ", host addr: " << mem_buf->host_ptr_
                << ", device addr: " << mem_buf->device_ptr_ << ", image_addr: " << image
                << ", total size: " << total_size_;
  return host_ptr;
}

void OpenCLAllocator::Free(void *buf) {
  if (buf == nullptr) {
    return;
  }
  Lock();
  auto iter = allocated_list_.find(buf);
  if (iter != allocated_list_.end()) {
    if (iter->second->map_flags_) {
      int ret = UnmapBuffer(buf);
      if (ret != RET_OK) {
        MS_LOG(WARNING) << "UnmapBuffer failed.";
      }
      iter->second->map_flags_ = false;
    }
    auto mem_buf = iter->second;
    allocated_list_.erase(iter);
    free_list_.insert(std::make_pair(mem_buf->size_, mem_buf));
    UnLock();
    MS_LOG(DEBUG) << "Free device buffer. size: " << mem_buf->size_ << ", host addr: " << mem_buf->host_ptr_
                  << ", device addr: " << mem_buf->device_ptr_ << ", image addr: " << mem_buf->image_ptr_
                  << ", free list size: " << free_list_.size();
    return;
  }
  UnLock();
  MS_LOG(WARNING) << "Host ptr " << buf << " has freed";
}

size_t OpenCLAllocator::total_size() {
  Lock();
  size_t totalSize = 0;

  for (auto it = allocated_list_.begin(); it != allocated_list_.end(); it++) {
    totalSize += it->second->size_;
  }

  for (auto it = free_list_.begin(); it != free_list_.end(); it++) {
    totalSize += it->second->size_;
  }
  UnLock();
  return totalSize;
}

void *OpenCLAllocator::GetImage(void *buffer) {
  auto it = allocated_list_.find(buffer);
  if (it != allocated_list_.end()) {
    return it->second->image_ptr_;
  }
  return nullptr;
}

void *OpenCLAllocator::GetBuffer(void *buffer) {
  auto it = allocated_list_.find(buffer);
  if (it != allocated_list_.end()) {
    return it->second->device_ptr_;
  }
  return nullptr;
}

template <typename T>
void OpenCLAllocator::ClearMemList(T *list) {
  auto svm_capabilities = ocl_runtime_->GetSVMCapabilities();
  for (auto it = list->begin(); it != list->end(); it++) {
    if (it->second->map_flags_) {
      int ret = UnmapBuffer(it->second->host_ptr_);
      if (ret != RET_OK) {
        MS_LOG(WARNING) << "UnmapBuffer failed.";
      }
    }
    if (svm_capabilities) {
      clSVMFree((*ocl_runtime_->Context())(), it->second->host_ptr_);
      MS_LOG(DEBUG) << "OpenCL free svm buffer : " << it->second->host_ptr_;
    } else {
      cl::Buffer *buffer = static_cast<cl::Buffer *>(it->second->device_ptr_);
      MS_LOG(DEBUG) << "OpenCL free device buffer : " << buffer;
      if (buffer != nullptr) {
        delete buffer;
        it->second->device_ptr_ = nullptr;
      }
      cl::Image *image = static_cast<cl::Image *>(it->second->image_ptr_);
      if (image != nullptr) {
        delete image;
        it->second->image_ptr_ = nullptr;
      }
    }
    delete it->second;
  }
  list->clear();
}

void OpenCLAllocator::Clear() {
  Lock();
  ClearMemList<std::unordered_map<void *, MemBuf *>>(&allocated_list_);
  ClearMemList<std::multimap<size_t, MemBuf *>>(&free_list_);
  UnLock();
}

void *OpenCLAllocator::MapBuffer(void *host_ptr, int flags, void *command_queue, bool sync) {
  auto svm_capabilities = ocl_runtime_->GetSVMCapabilities();
  if (svm_capabilities) {
    if (!(svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
      auto it = allocated_list_.find(host_ptr);
      if (it == allocated_list_.end()) {
        MS_LOG(ERROR) << "Map buffer failed, can not found buffer :" << host_ptr;
        return nullptr;
      }
      int ret = ocl_runtime_->MapBuffer(host_ptr, flags, it->second->size_,
                                        static_cast<cl::CommandQueue *>(command_queue), sync);
      if (ret != RET_OK) {
        MS_LOG(WARNING) << "MapBuffer failed.";
      }
    }
    return host_ptr;
  }
  Lock();
  auto it = allocated_list_.find(host_ptr);
  if (it == allocated_list_.end()) {
    UnLock();
    MS_LOG(ERROR) << "Map buffer failed, can not found buffer :" << host_ptr;
    return nullptr;
  }

  if (it->second->map_flags_) {
    UnLock();
    MS_LOG(WARNING) << "Host ptr " << host_ptr << " has mapped";
    return host_ptr;
  }
  MemBuf *mem_buf = it->second;
  MS_ASSERT(mem_buf);
  void *new_host_ptr{nullptr};
  if (mem_buf->mem_type_ == MemType::BUF) {
    cl::Buffer *buffer = static_cast<cl::Buffer *>(mem_buf->device_ptr_);
    MS_ASSERT(buffer);
    new_host_ptr = ocl_runtime_->MapBuffer(*buffer, flags, mem_buf->size_, nullptr, sync);
  } else {
    std::vector<size_t> region{mem_buf->img_size_.width, mem_buf->img_size_.height, 1};
    cl::Image2D *image = static_cast<cl::Image2D *>(mem_buf->image_ptr_);
    MS_ASSERT(image);
    new_host_ptr = ocl_runtime_->MapBuffer(*image, sync, CL_MAP_READ | CL_MAP_WRITE, region);
  }
  if (new_host_ptr == nullptr) {
    UnLock();
    MS_LOG(WARNING) << "Map buffer failed, can not found buffer or already mapped, dev_ptr=" << mem_buf->device_ptr_
                    << ", host_ptr=" << host_ptr;
    return nullptr;
  }

  mem_buf->map_flags_ = true;
  mem_buf->host_ptr_ = new_host_ptr;
  allocated_list_.erase(it);
  allocated_list_[new_host_ptr] = mem_buf;
  UnLock();
  MS_LOG(DEBUG) << "Map buffer form " << host_ptr << " to " << new_host_ptr;
  return new_host_ptr;
}

int OpenCLAllocator::UnmapBuffer(void *host_ptr, void *command_queue) {
  auto svm_capabilities = ocl_runtime_->GetSVMCapabilities();
  if (svm_capabilities) {
    if (!(svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
      return ocl_runtime_->UnmapBuffer(host_ptr);
    }
    return RET_OK;
  }
  auto it = allocated_list_.find(host_ptr);
  if (it == allocated_list_.end()) {
    MS_LOG(ERROR) << "Map buffer failed, can not found buffer :" << host_ptr;
    return RET_ERROR;
  }
  if (it->second->map_flags_) {
    it->second->map_flags_ = false;
    cl::Memory *mem = static_cast<cl::Memory *>(it->second->mem_type_ == MemType::BUF ? it->second->device_ptr_
                                                                                      : it->second->image_ptr_);
    return ocl_runtime_->UnmapBuffer(*mem, it->second->host_ptr_, static_cast<cl::CommandQueue *>(command_queue));
  } else {
    MS_LOG(WARNING) << "Host ptr " << host_ptr << " do not mapped";
    return RET_OK;
  }
}

MemType OpenCLAllocator::GetMemType(void *host_ptr) {
  MemType mem_type{MemType::BUF};
  Lock();
  auto it = allocated_list_.find(host_ptr);
  if (it == allocated_list_.end()) {
    UnLock();
    MS_LOG(ERROR) << "Can not found buffer :" << host_ptr;
    return mem_type;
  }
  MemBuf *mem_buf = it->second;
  MS_ASSERT(mem_buf);
  mem_type = mem_buf->mem_type_;
  UnLock();
  return mem_type;
}

int OpenCLAllocator::GetImageSize(void *host_ptr, ImageSize *img_size) {
  MS_ASSERT(img_size);
  Lock();
  auto it = allocated_list_.find(host_ptr);
  if (it == allocated_list_.end()) {
    UnLock();
    MS_LOG(ERROR) << "Can not found buffer :" << host_ptr;
    return RET_OK;
  }
  MemBuf *mem_buf = it->second;
  MS_ASSERT(mem_buf);
  if (mem_buf->mem_type_ == MemType::IMG) {
    *img_size = mem_buf->img_size_;
  }
  UnLock();
  return RET_OK;
}
}  // namespace mindspore::lite::opencl
