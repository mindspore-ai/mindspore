/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
package com.mindspore.config;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.logging.Logger;
import java.util.Locale;

/**
 * NativeLibrary Class
 *
 * @since v1.0
 */
public class NativeLibrary {
    private static final Logger LOGGER = Logger.getLogger(NativeLibrary.class.toString());

    private static final String GLOG_LIBNAME = "mindspore_glog";
    private static final String MINDSPORE_CORE_LIBNAME = "mindspore_core";
    private static final String OPENCV_CORE_LIBNAME = "opencv_core";
    private static final String OPENCV_IMGCODECS_LIBNAME = "opencv_imgcodecs";
    private static final String OPENCV_IMGPROC_LIBNAME = "opencv_imgproc";
    private static final String MSLITE_CONVERTER_PLUGIN_LIBNAME = "mslite_converter_plugin";
    private static final String MINDSPORE_CONVERTER_LIBNAME = "mindspore_converter";
    private static final String MINDSPORE_LITE_LIBNAME = "mindspore-lite";
    private static final String MSPLUGIN_GE_LITERT_LIBNAME = "msplugin-ge-litert";
    private static final String RUNTIME_CONVERT_PLUGIN_LIBNAME = "runtime_convert_plugin";
    private static final String DNNL_LIBNAME = "dnnl";
    private static final String LITE_UNIFIED_EXECUTOR_LIBNAME = "lite-unified-executor";
    private static final String MINDSPORE_LITE_JNI_LIBNAME = "mindspore-lite-jni";
    private static final String MINDSPORE_LITE_TRAIN_LIBNAME = "mindspore-lite-train";
    private static final String MINDSPORE_LITE_TRAIN_JNI_LIBNAME = "mindspore-lite-train-jni";

    /**
     * Load function.
     */
    public static void load() {
        if (isLibLoaded() || loadLibrary()) {
            LOGGER.info("Native lib has been loaded.");
            return;
        }
        loadLibs();
    }

    /**
     * Load native libs function.
     *
     * dynamic library as follows:
     * libmindspore_glog.so
     * libopencv_core.so
     * libopencv_imgproc.so
     * libopencv_imgcodecs.so
     * libmslite_converter_plugin.so
     * libmindspore_core.so
     * libmindspore_converter.so
     * libmindspore-lite.so
     * mindspore-lite-jni
     *
     * For cloud inference, dlopen library as follows:
     * libmsplugin-ge-litert
     * libruntime_convert_plugin
     */
    private static void loadLibs() {
        loadLib(makeResourceName("lib" + GLOG_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + OPENCV_CORE_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + OPENCV_IMGPROC_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + OPENCV_IMGCODECS_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_CORE_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MSLITE_CONVERTER_PLUGIN_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_CONVERTER_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + DNNL_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + LITE_UNIFIED_EXECUTOR_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_LITE_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_LITE_JNI_LIBNAME + ".so"));
    }

    private static boolean isLibLoaded() {
        try {
            Version.version();
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
        return true;
    }

    /**
     * Load library function.
     * If any jni lib is loaded successfully, the function return True.
     * jni lib: mindspore-lite-jni, mindspore-lite-train-jni
     */
    private static boolean loadLibrary() {
        boolean loadSuccess = false;
        try {
            System.loadLibrary(MINDSPORE_LITE_JNI_LIBNAME);
            loadSuccess = true;
            LOGGER.info("loadLibrary mindspore-lite-jni success");
        } catch (UnsatisfiedLinkError e) {
            LOGGER.info(String.format(Locale.ENGLISH, "tryLoadLibrary mindspore-lite-jni failed: %s", e));
        }
        try {
            System.loadLibrary(MINDSPORE_LITE_TRAIN_JNI_LIBNAME);
            loadSuccess = true;
            LOGGER.info("loadLibrary mindspore-lite-train-jni success.");
        } catch (UnsatisfiedLinkError e) {
            LOGGER.info(String.format(Locale.ENGLISH, "tryLoadLibrary mindspore-lite-train-jni failed: %s", e));
        }
        return loadSuccess;
    }

    private static void loadLib(String libResourceName) {
        LOGGER.info(String.format(Locale.ENGLISH,"start load libResourceName: %s.", libResourceName));
        final InputStream libResource = NativeLibrary.class.getClassLoader().getResourceAsStream(libResourceName);
        if (libResource == null) {
            LOGGER.warning(String.format(Locale.ENGLISH,"lib file: %s not exist.", libResourceName));
            return;
        }
        try {
            final File tmpDir = mkTmpDir();
            String libName = libResourceName.substring(libResourceName.lastIndexOf('/') + 1);
            tmpDir.deleteOnExit();

            // copy file to tmpFile
            final File tmpFile = new File(tmpDir.getCanonicalPath(), libName);
            tmpFile.deleteOnExit();
            LOGGER.info(String.format(Locale.ENGLISH,"extract %d bytes to %s", copyLib(libResource, tmpFile),
                    tmpFile));
            if (("lib" + MINDSPORE_LITE_LIBNAME + ".so").equals(libName)) {
                extractLib(makeResourceName("lib" + MSPLUGIN_GE_LITERT_LIBNAME + ".so"), tmpDir);
                extractLib(makeResourceName("lib" + RUNTIME_CONVERT_PLUGIN_LIBNAME + ".so"), tmpDir);
            }
            System.load(tmpFile.toString());
        } catch (IOException e) {
            throw new UnsatisfiedLinkError(
                    String.format(Locale.ENGLISH,"extract library into tmp file (%s) failed.", e));
        }
    }

    private static long copyLib(InputStream libResource, File tmpFile) throws IOException {
        try (FileOutputStream outputStream = new FileOutputStream(tmpFile);) {
            // 1MB
            byte[] buffer = new byte[1 << 20];
            long byteCnt = 0;
            int n = 0;
            while ((n = libResource.read(buffer)) >= 0) {
                outputStream.write(buffer, 0, n);
                byteCnt += n;
            }
            return byteCnt;
        } finally {
            libResource.close();
        }
    }


    private static File mkTmpDir() {
        Long timestamp = System.currentTimeMillis();
        String dirName = "mindspore_lite_libs-" + timestamp + "-";
        for (int i = 0; i < 10; i++) {
            File tmpDir = new File(new File(System.getProperty("java.io.tmpdir")), dirName + i);
            if (tmpDir.mkdir()) {
                return tmpDir;
            }
        }
        throw new IllegalStateException("create tmp dir failed, dirName: " + dirName);
    }

    /**
     * get native lib file name, eg. com/mindspore/lite/linux_x86_64/libmindspore-lite.so
     *
     * @param basename
     * @return
     */
    private static String makeResourceName(String basename) {
        return "com/mindspore/lite/" + String.format("linux_%s/", architecture()) + basename;
    }

    private static String architecture() {
        final String arch = System.getProperty("os.arch").toLowerCase();
        return ("amd64".equals(arch)) ? "x86_64" : arch;
    }

    private static void extractLib(String libResourceName, File targetDir) {
        try {
            final InputStream dependLibRes = NativeLibrary.class.getClassLoader().getResourceAsStream(libResourceName);
            if (dependLibRes == null) {
                LOGGER.warning(String.format("lib file: %s not exist.", libResourceName));
                return;
            }
            String dependLibName = libResourceName.substring(libResourceName.lastIndexOf("/") + 1);
            final File tmpDependFile = new File(targetDir.getCanonicalPath(), dependLibName);
            tmpDependFile.deleteOnExit();
            LOGGER.info(String.format("extract %d bytes to %s", copyLib(dependLibRes, tmpDependFile), tmpDependFile));
        } catch (IOException e) {
            LOGGER.warning(String.format("extract library into tmp file (%s) failed.", e.toString()));
        }
    }
}
