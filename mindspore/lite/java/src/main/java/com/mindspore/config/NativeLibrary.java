package com.mindspore.config;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.logging.Logger;

public class NativeLibrary {
    private static final Logger LOGGER = MindsporeLite.GetLogger();

    private static final String GLOG_LIBNAME = "mindspore_glog";
    private static final String JPEG_LIBNAME = "jpeg";
    private static final String TURBOJPEG_LIBNAME = "turbojpeg";
    private static final String MINDDATA_LITE_LIBNAME = "minddata-lite";
    private static final String MINDSPORE_LITE_LIBNAME = "mindspore-lite";
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
     */
    public static void loadLibs() {
        loadLib(makeResourceName("lib" + GLOG_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + JPEG_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + TURBOJPEG_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDDATA_LITE_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_LITE_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_LITE_JNI_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_LITE_TRAIN_LIBNAME + ".so"));
        loadLib(makeResourceName("lib" + MINDSPORE_LITE_TRAIN_JNI_LIBNAME + ".so"));
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
            LOGGER.info("loadLibrary " + MINDSPORE_LITE_JNI_LIBNAME + ": success");
        } catch (UnsatisfiedLinkError e) {
            LOGGER.info("tryLoadLibrary " + MINDSPORE_LITE_JNI_LIBNAME + " failed.");
        }
        try {
            System.loadLibrary(MINDSPORE_LITE_TRAIN_JNI_LIBNAME);
            loadSuccess = true;
            LOGGER.info("loadLibrary " + MINDSPORE_LITE_TRAIN_JNI_LIBNAME + ": success.");
        } catch (UnsatisfiedLinkError e) {
            LOGGER.info("tryLoadLibrary " + MINDSPORE_LITE_TRAIN_JNI_LIBNAME + " failed.");
        }
        return loadSuccess;
    }

    private static void loadLib(String libResourceName) {
        LOGGER.info("start load libResourceName: " + libResourceName);
        final InputStream libResource = NativeLibrary.class.getClassLoader().getResourceAsStream(libResourceName);
        if (libResource == null) {
            LOGGER.warning(String.format("lib file: %s not exist.", libResourceName));
            return;
        }
        try {
            final File tmpDir = mkTmpDir();
            String libName = libResourceName.substring(libResourceName.lastIndexOf("/") + 1);
            tmpDir.deleteOnExit();

            //copy file to tmpFile
            final File tmpFile = new File(tmpDir.getCanonicalPath(), libName);
            tmpFile.deleteOnExit();
            LOGGER.info(String.format("extract %d bytes to %s", copyLib(libResource, tmpFile), tmpFile));
            System.load(tmpFile.toString());
        } catch (IOException e) {
            throw new UnsatisfiedLinkError(
                    String.format("extract library into tmp file (%s) failed.", e.toString()));
        }
    }

    private static long copyLib(InputStream libResource, File tmpFile) throws IOException{
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
        final String MINDSPORE_LITE_LIBS = "mindspore_lite_libs-";
        Long timestamp = System.currentTimeMillis();
        String dirName = MINDSPORE_LITE_LIBS + timestamp + "-";
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
        return (arch.equals("amd64")) ? "x86_64" : arch;
    }
}
