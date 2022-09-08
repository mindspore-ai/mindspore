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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_OPENGL_UTIL_H_
#define MINDSPORE_LITE_TOOLS_COMMON_OPENGL_UTIL_H_

#include <map>
#include <vector>
#include <utility>
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

#if defined(GPU_OPENCL) && defined(__ANDROID__) && defined(ENABLE_ARM64)
#include "EGL/egl.h"
#include "GLES3/gl3.h"
#include "GLES3/gl32.h"
#else
typedef unsigned int GLuint;
typedef unsigned int GLenum;
typedef signed int khronos_ssize_t;
typedef khronos_ssize_t GLsizeiptr;
#define GL_RGBA32F 0x8814
#define GL_TEXTURE_2D 0x0DE1
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_NONE 0x00
typedef void *EGLContext;
typedef void *EGLSurface;
typedef void *EGLDisplay;

inline EGLContext eglGetCurrentContext(void) { return nullptr; }
inline EGLContext eglGetCurrentDisplay(void) { return nullptr; }
#endif

#define OPEN_GL_CHECK_ERROR

#ifdef OPEN_GL_CHECK_ERROR
#define OPENGL_CHECK_ERROR                                  \
  {                                                         \
    GLenum error = glGetError();                            \
    if (GL_NO_ERROR != error) {                             \
      MS_LOG(ERROR) << "ERR CHECK Fail, error = " << error; \
      return 0;                                             \
    }                                                       \
    MS_ASSERT(GL_NO_ERROR == error);                        \
  }
#else
#define OPENGL_CHECK_ERROR
#endif

namespace mindspore {
namespace OpenGL {
#define BIND_INDEX_0 0
#define BIND_INDEX_1 1
#define BIND_INDEX_2 2
#define BIND_INDEX_3 3
#define BIND_INDEX_4 4

class OpenGLRuntime {
 public:
  OpenGLRuntime() {}
  virtual ~OpenGLRuntime() {}

  bool Init();
  GLuint GLCreateTexture(int w, int h, int c, GLenum textrueFormat = GL_RGBA32F, GLenum target = GL_TEXTURE_2D);

  // GLuint CopyHostToDeviceTexture(void *hostData, int width, int height);
  GLuint CopyHostToDeviceTexture(void *hostData, int width, int height, int channel);
  void *CopyDeviceTextureToHost(GLuint textureID);

  void PrintImage2DData(float *data, int w, int h, int c = 4);

 private:
  static GLuint LoadShader(GLenum shaderType, const char *pSource);
  static GLuint CreateComputeProgram(const char *pComputeSource);
  GLuint GLCreateSSBO(GLsizeiptr size, void *hostData = nullptr, GLenum type = GL_SHADER_STORAGE_BUFFER,
                      GLenum usage = GL_DYNAMIC_DRAW);
  bool CopyDeviceSSBOToHost(GLuint ssboBufferID, void *hostData, GLsizeiptr size);
  bool CopyHostToDeviceSSBO(void *hostData, GLuint ssboBufferID, GLsizeiptr size);

  bool CopyDeviceTextureToSSBO(GLuint textureID, GLuint ssboBufferID);
  bool CopyDeviceSSBOToTexture(GLuint ssboBufferID, GLuint textureID);

  std::map<GLuint, std::pair<GLsizeiptr, GLenum>> m_ssbo_pool_;
  std::map<GLuint, std::pair<std::vector<int>, std::vector<GLenum>>> m_texture_pool_;
  EGLContext m_context_ = nullptr;
  EGLDisplay m_display_ = nullptr;
  EGLSurface m_surface_ = nullptr;
};
}  // namespace OpenGL
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_COMMON_OPENGL_UTIL_H_
