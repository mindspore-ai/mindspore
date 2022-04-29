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

#include "tools/common/opengl_util.h"
#include <cstdlib>
#include <algorithm>

namespace mindspore {
namespace OpenGL {
#if defined(GPU_OPENCL) && defined(__ANDROID__) && defined(ENABLE_ARM64)
const char *g_glsl_host_to_device_2d =
  "#version 320 es\n"
  "#define PRECISION highp\n"
  "precision PRECISION float;\n"
  "#define FORMAT rgba32f\n"
  "layout(FORMAT, binding=0) writeonly uniform PRECISION image2D uImage;\n"
  "layout(binding=1) readonly buffer SSBO {\n"
  "    float data[];\n"
  "} uInBuffer;\n"
  "layout(location = 2) uniform int uWidth;\n"
  "layout(location = 3) uniform int uHeight;\n"
  "layout(location = 4) uniform int uChannel;\n"
  "layout (local_size_x = 4, local_size_y = 4, local_size_z = 1) in;\n"
  "void main()\n"
  "{\n"
  "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
  "    if (pos.x < uWidth && pos.y < uHeight)\n"
  "    {\n"
  "        vec4 color;\n"
  "        color.r = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + 0];\n"
  "        color.g = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + 1];\n"
  "        color.b = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + 2];\n"
  "        color.a = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + 3];\n"
  "        imageStore(uImage, pos.xy, color);\n"
  "    }\n"
  "}\n";

const char *g_glsl_host_to_device_3d =
  "#version 320 es\n"
  "#define PRECISION highp\n"
  "precision PRECISION float;\n"
  "#define FORMAT rgba32f\n"
  "layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uImage;\n"
  "layout(binding=1) readonly buffer SSBO {\n"
  "    float data[];\n"
  "} uInBuffer;\n"
  "layout(location = 2) uniform int uWidth;\n"
  "layout(location = 3) uniform int uHeight;\n"
  "layout(location = 4) uniform int uChannel;\n"
  "layout (local_size_x = 4, local_size_y = 4, local_size_z = 1) in;\n"
  "void main()\n"
  "{\n"
  "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
  "    if (pos.x < uWidth && pos.y < uHeight)\n"
  "    {\n"
  "        vec4 color;\n"
  "        int z = pos.z*4;\n"
  "        color.r = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + (z+0)];\n"
  "        color.g = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + (z+1)];\n"
  "        color.b = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + (z+2)];\n"
  "        color.a = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + (z+3)];\n"
  "        imageStore(uImage, pos, color);\n"
  "    }\n"
  "}\n";

const char *g_glsl_device_to_host_2d =
  "#version 320 es\n"
  "#define PRECISION highp\n"
  "precision PRECISION float;\n"
  "#define FORMAT rgba32f\n"
  "layout(FORMAT, binding=0) readonly uniform PRECISION image2D uImage;\n"
  "layout(binding=1) writeonly buffer destBuffer{\n"
  "    float data[];\n"
  "} uOutBuffer;\n"
  "layout(location = 2) uniform int uWidth;\n"
  "layout(location = 3) uniform int uHeight;\n"
  "layout(location = 4) uniform int uChannel;\n"
  "layout (local_size_x = 4, local_size_y = 4, local_size_z = 1) in;\n"
  "#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n"
  "void main()\n"
  "{\n"
  "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
  "    if (pos.x < uWidth  && pos.y < uHeight)\n"
  "    {\n"
  "        vec4 color = imageLoad(uImage, ivec2(pos.x * UP_DIV(uChannel, 4) + pos.z, pos.y));\n"
  "        int z = pos.z*4;\n"
  "         for (int i = 0; i < 4; i++) {\n"
  "            if (z + i < uChannel) {\n"
  "              uOutBuffer.data[pos.y*uWidth*uChannel+pos.x*uChannel+(z+i)] = color[i];\n"
  "            } \n"
  "         } \n"
  "    }\n"
  "}\n";

const char *g_glsl_device_to_host_3d =
  "#version 320 es\n"
  "#define PRECISION highp\n"
  "precision PRECISION float;\n"
  "#define FORMAT rgba32f\n"
  "layout(FORMAT, binding=0) readonly uniform PRECISION image3D uImage;\n"
  "layout(binding=1) writeonly buffer destBuffer{\n"
  "    float data[];\n"
  "} uOutBuffer;\n"
  "layout(location = 2) uniform int uWidth;\n"
  "layout(location = 3) uniform int uHeight;\n"
  "layout(location = 4) uniform int uChannel;\n"
  "layout (local_size_x = 4, local_size_y = 4, local_size_z = 1) in;\n"
  "void main()\n"
  "{\n"
  "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
  "    if (pos.x < uWidth && pos.y < uHeight)\n"
  "    {\n"
  "        vec4 color = imageLoad(uImage, pos);\n"
  "        int z = pos.z*4;\n"
  "        uOutBuffer.data[pos.y*uWidth*uChannel+pos.x*uChannel+(z+0)] = color.r;\n"
  "        uOutBuffer.data[pos.y*uWidth*uChannel+pos.x*uChannel+(z+1)] = color.g;\n"
  "        uOutBuffer.data[pos.y*uWidth*uChannel+pos.x*uChannel+(z+2)] = color.b;\n"
  "        uOutBuffer.data[pos.y*uWidth*uChannel+pos.x*uChannel+(z+3)] = color.a;\n"
  "    }\n"
  "}\n";

constexpr int kC4Align = 4;
constexpr int kWidthIndex = 0;
constexpr int kHeightIndex = 1;
constexpr int kChannelIndex = 2;

bool OpenGLRuntime::Init() {
  MS_LOG(INFO) << "Rt Init Begin";
  if (!(eglGetCurrentContext() != EGL_NO_CONTEXT)) {
    m_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (m_display_ == EGL_NO_DISPLAY) {
      MS_LOG(ERROR) << "eglGetDisplay error";
      return false;
    }

    int majorVersion;
    int minorVersion;
    auto glRet = eglInitialize(m_display_, &majorVersion, &minorVersion);
    if (glRet != EGL_TRUE) {
      MS_LOG(ERROR) << "eglInitialize error";
      return false;
    }

    EGLint numConfigs;
    static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                           EGL_PBUFFER_BIT,
                                           EGL_RENDERABLE_TYPE,
                                           EGL_OPENGL_ES2_BIT,
                                           EGL_RED_SIZE,
                                           8,
                                           EGL_GREEN_SIZE,
                                           8,
                                           EGL_BLUE_SIZE,
                                           8,
                                           EGL_ALPHA_SIZE,
                                           8,
                                           EGL_NONE};

    EGLConfig surfaceConfig;
    if (!eglChooseConfig(m_display_, configAttribs, &surfaceConfig, 1, &numConfigs)) {
      eglMakeCurrent(m_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglTerminate(m_display_);
      m_display_ = EGL_NO_DISPLAY;
      MS_LOG(ERROR) << "eglChooseConfig error";
      return false;
    }

    static const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    m_context_ = eglCreateContext(m_display_, surfaceConfig, NULL, contextAttribs);
    static const EGLint surfaceAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    m_surface_ = eglCreatePbufferSurface(m_display_, surfaceConfig, surfaceAttribs);

    glRet = eglMakeCurrent(m_display_, m_surface_, m_surface_, m_context_);
    if (glRet != EGL_TRUE) {
      MS_LOG(ERROR) << "eglMakeCurrent error";
      return false;
    }

    eglBindAPI(EGL_OPENGL_ES_API);
  } else {
    m_context_ = EGL_NO_CONTEXT;
    MS_LOG(ERROR) << "eglGetCurrentContext() != EGL_NO_CONTEXT";
    return false;
  }
  MS_LOG(INFO) << "Rt Init End";
  return true;
}

GLuint OpenGLRuntime::LoadShader(GLenum shaderType, const char *pSource) {
  GLuint shader = glCreateShader(shaderType);
  if (shader) {
    glShaderSource(shader, 1, &pSource, NULL);
    glCompileShader(shader);
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
      GLint infoLen = 0;
      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
      if (infoLen > 0) {
        char *buf = reinterpret_cast<char *>(malloc(infoLen));
        MS_CHECK_TRUE_MSG(buf != nullptr, 0, "Malloc OpenGL Buffer failed");
        glGetShaderInfoLog(shader, infoLen, NULL, buf);
        fprintf(stderr, "Could not compile shader %d:\n%s\n", shaderType, buf);
        free(buf);
        glDeleteShader(shader);
        shader = 0;
      }
    }
  }
  return shader;
}

GLuint OpenGLRuntime::CreateComputeProgram(const char *pComputeSource) {
  GLuint computeShader = LoadShader(GL_COMPUTE_SHADER, pComputeSource);
  if (!computeShader) {
    return 0;
  }

  GLuint program = glCreateProgram();
  if (program) {
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE) {
      GLint bufLength = 0;
      glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
      if (bufLength > 0) {
        char *buf = reinterpret_cast<char *>(malloc(bufLength));
        MS_CHECK_TRUE_MSG(buf != nullptr, 0, "Malloc OpenGL Buffer failed");
        glGetProgramInfoLog(program, bufLength, NULL, buf);
        fprintf(stderr, "Could not link program:\n%s\n", buf);
        free(buf);
      }
      glDeleteProgram(program);
      program = 0;
    }
  }
  return program;
}

GLuint OpenGLRuntime::GLCreateSSBO(GLsizeiptr size, void *hostData, GLenum type, GLenum usage) {
  MS_ASSERT(size > 0);

  GLuint ssboBufferID;
  glGenBuffers(1, &ssboBufferID);
  OPENGL_CHECK_ERROR;

  glBindBuffer(type, ssboBufferID);
  OPENGL_CHECK_ERROR;
  MS_ASSERT(ssboBufferID > 0);

  glBufferData(type, size, hostData, usage);
  OPENGL_CHECK_ERROR;

  MS_ASSERT(m_ssbo_pool_.count(ssboBufferID) == 0);
  m_ssbo_pool_[ssboBufferID] = std::make_pair(size, type);

  return ssboBufferID;
}

bool OpenGLRuntime::CopyDeviceSSBOToHost(GLuint ssboBufferID, void *hostData, GLsizeiptr size) {
  MS_ASSERT(m_ssbo_pool_.count(ssboBufferID) > 0);
  MS_ASSERT(m_ssbo_pool_[ssboBufferID].first >= size);

  glBindBuffer(m_ssbo_pool_[ssboBufferID].second, ssboBufferID);
  OPENGL_CHECK_ERROR;

  auto ptr = glMapBufferRange(m_ssbo_pool_[ssboBufferID].second, 0, m_ssbo_pool_[ssboBufferID].first, GL_MAP_READ_BIT);
  OPENGL_CHECK_ERROR;

  if (ptr != nullptr) {
    ::memcpy(hostData, ptr, size);
  }

  glUnmapBuffer(m_ssbo_pool_[ssboBufferID].second);
  OPENGL_CHECK_ERROR;
  return true;
}

bool OpenGLRuntime::CopyHostToDeviceSSBO(void *hostData, GLuint ssboBufferID, GLsizeiptr size) {
  MS_ASSERT(m_ssbo_pool_.count(ssboBufferID) > 0);
  MS_ASSERT(m_ssbo_pool_[ssboBufferID].first >= size);

  glBindBuffer(m_ssbo_pool_[ssboBufferID].second, ssboBufferID);
  OPENGL_CHECK_ERROR;

  auto ptr = glMapBufferRange(m_ssbo_pool_[ssboBufferID].second, 0, m_ssbo_pool_[ssboBufferID].first, GL_MAP_READ_BIT);
  OPENGL_CHECK_ERROR;

  if (ptr != nullptr) {
    ::memcpy(ptr, hostData, size);
  }

  glUnmapBuffer(m_ssbo_pool_[ssboBufferID].second);
  OPENGL_CHECK_ERROR;
  return true;
}

GLuint OpenGLRuntime::GLCreateTexture(int w, int h, int c, GLenum TextrueFormat, GLenum target) {
  GLuint textureID = 0;
  if (target == GL_TEXTURE_3D) {
    MS_ASSERT(w > 0 && h > 0 && c > 0);
    glGenTextures(1, &textureID);
    OPENGL_CHECK_ERROR;
    glBindTexture(target, textureID);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    OPENGL_CHECK_ERROR;

    int realW = w;
    int realH = h;
    int realD = UP_DIV(c, kC4Align);
    glTexStorage3D(target, 1, TextrueFormat, realW, realH, realD);
    OPENGL_CHECK_ERROR;
  } else if (target == GL_TEXTURE_2D) {
    MS_ASSERT(w > 0 && h > 0);
    glGenTextures(1, &textureID);
    OPENGL_CHECK_ERROR;
    glBindTexture(target, textureID);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    OPENGL_CHECK_ERROR;
    glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    OPENGL_CHECK_ERROR;

    int realW = w * UP_DIV(c, kC4Align);
    int realH = h;
    glTexStorage2D(target, 1, TextrueFormat, realW, realH);
    OPENGL_CHECK_ERROR;
  }

  std::vector<int> dims = {w, h, c};
  std::vector<GLenum> props = {TextrueFormat, target};
  m_texture_pool_[textureID] = std::make_pair(dims, props);

  return textureID;
}

bool OpenGLRuntime::CopyDeviceTextureToSSBO(GLuint textureID, GLuint ssboBufferID) {
  if (m_texture_pool_.find(textureID) == m_texture_pool_.end()) {
    return false;
  }
  GLuint computeProgram;
  if (m_texture_pool_[textureID].second[1] == GL_TEXTURE_2D) {
    computeProgram = OpenGLRuntime::CreateComputeProgram(g_glsl_device_to_host_2d);
  } else {
    computeProgram = OpenGLRuntime::CreateComputeProgram(g_glsl_device_to_host_3d);
  }
  glUseProgram(computeProgram);

  // bind the src image texture
  glBindImageTexture(BIND_INDEX_0, textureID, 0, GL_TRUE, 0, GL_READ_ONLY, GL_RGBA32F);

  // bind the dest output data
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDEX_1, ssboBufferID);

  // set uniform values
  int width = m_texture_pool_[textureID].first[kWidthIndex];
  int height = m_texture_pool_[textureID].first[kHeightIndex];
  int channel = m_texture_pool_[textureID].first[kChannelIndex];

  glUniform1i(BIND_INDEX_2, width);
  glUniform1i(BIND_INDEX_3, height);
  glUniform1i(BIND_INDEX_4, channel);

  int c_4 = UP_DIV(channel, kC4Align);
  int gLocalSize[3] = {4, 4, 1};
  glDispatchCompute(UP_DIV(width, gLocalSize[FIRST_INPUT]), UP_DIV(height, gLocalSize[SECOND_INPUT]),
                    UP_DIV(c_4, gLocalSize[THIRD_INPUT]));

  // memory sync
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glDeleteProgram(computeProgram);
  return true;
}

bool OpenGLRuntime::CopyDeviceSSBOToTexture(GLuint ssboBufferID, GLuint textureID) {
  if (m_texture_pool_.find(textureID) == m_texture_pool_.end()) {
    return false;
  }
  GLuint computeProgram;
  if (m_texture_pool_[textureID].second[1] == GL_TEXTURE_2D) {
    computeProgram = OpenGLRuntime::CreateComputeProgram(g_glsl_host_to_device_2d);
  } else {
    computeProgram = OpenGLRuntime::CreateComputeProgram(g_glsl_host_to_device_3d);
  }
  glUseProgram(computeProgram);

  // bind the src image texture
  glBindImageTexture(BIND_INDEX_0, textureID, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

  // bind the dest output data
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDEX_1, ssboBufferID);

  // set uniform values
  int width = m_texture_pool_[textureID].first[kWidthIndex];
  int height = m_texture_pool_[textureID].first[kHeightIndex];
  int channel = m_texture_pool_[textureID].first[kChannelIndex];

  glUniform1i(BIND_INDEX_2, width);
  glUniform1i(BIND_INDEX_3, height);
  glUniform1i(BIND_INDEX_4, channel);

  int c_4 = UP_DIV(channel, kC4Align);
  int gLocalSize[3] = {4, 4, 1};
  glDispatchCompute(UP_DIV(width, gLocalSize[FIRST_INPUT]), UP_DIV(height, gLocalSize[SECOND_INPUT]),
                    UP_DIV(c_4, gLocalSize[THIRD_INPUT]));
  // memory sync
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glDeleteProgram(computeProgram);
  return true;
}

GLuint OpenGLRuntime::CopyHostToDeviceTexture(void *hostData, int width, int height, int channel) {
  auto ssboBufferID = GLCreateSSBO(sizeof(float) * width * height * channel, hostData);
  auto textureID = GLCreateTexture(width, height, channel, GL_RGBA32F, GL_TEXTURE_2D);
  if (textureID == 0) {
    MS_LOG(ERROR) << "generate GlTexture failed";
  }
  CopyDeviceSSBOToTexture(ssboBufferID, textureID);
  return textureID;
}

void *OpenGLRuntime::CopyDeviceTextureToHost(GLuint textureID) {
  int width = m_texture_pool_[textureID].first[kWidthIndex];
  int height = m_texture_pool_[textureID].first[kHeightIndex];
  int channel = m_texture_pool_[textureID].first[kChannelIndex];

  auto ssboBufferID = GLCreateSSBO(sizeof(float) * width * height * channel);
  CopyDeviceTextureToSSBO(textureID, ssboBufferID);
  void *output = malloc(sizeof(float) * width * height * channel);
  if (output == nullptr) {
    MS_LOG(ERROR) << "Malloc host data failed";
    return nullptr;
  }
  CopyDeviceSSBOToHost(ssboBufferID, output, sizeof(float) * width * height * channel);
  return output;
}

void OpenGLRuntime::PrintImage2DData(float *data, int w, int h, int c) {
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      for (int k = 0; k < c; k++) {
        std::cout << data[w * c * i + c * j + k] << " ";
      }
      std::cout << "    ";
    }
  }
  std::cout << "data print finish!" << std::endl;
}
#else
bool OpenGLRuntime::Init() {
  MS_LOG(ERROR) << "Init error, server benchmark don't support opengl";
  return false;
}

GLuint OpenGLRuntime::GLCreateTexture(int w, int h, int c, GLenum TextrueFormat, GLenum target) { return 0; }
void *OpenGLRuntime::CopyDeviceTextureToHost(GLuint textureID) { return nullptr; }
GLuint OpenGLRuntime::CopyHostToDeviceTexture(void *hostData, int width, int height, int channel) { return 0; }

void OpenGLRuntime::PrintImage2DData(float *data, int w, int h, int c) {}
#endif
}  // namespace OpenGL
}  // namespace mindspore
