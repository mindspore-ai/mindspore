package com.mindspore.lite;


import java.io.*;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.io.IOUtils;

public class NativeLibrary {
    private static final Logger LOGGER = Logger.getLogger(NativeLibrary.class.toString());

    private static final String GLOG_LIBNAME = "glog";
    private static final String MINDSPORE_LITE_LIBNAME = "mindspore-lite";
    private static final String MINDSPORE_LITE_JNI_LIBNAME = "mindspore-lite-jni";

    private static final String MINDSPORE_LITE_LIBS = "mindspore_lite_libs";

    public static void load() {
        if (isLoaded()) {
            return;
        }
        loadLib(makeResourceDir(MINDSPORE_LITE_LIBS), makeResourceName("lib" + GLOG_LIBNAME + ".so.0"));
        loadLib(makeResourceDir(MINDSPORE_LITE_LIBS), makeResourceName("lib" + MINDSPORE_LITE_LIBNAME + ".so"));
        loadLib(makeResourceDir(MINDSPORE_LITE_LIBS), makeResourceName("lib" + MINDSPORE_LITE_JNI_LIBNAME + ".so"));
    }

    private static boolean tryLoadLibrary() {
        try {
            System.loadLibrary(MINDSPORE_LITE_LIBNAME);
            System.loadLibrary(MINDSPORE_LITE_JNI_LIBNAME);
            LOGGER.info("LoadLibrary: success");
            return true;
        } catch (UnsatisfiedLinkError e) {
            LOGGER.warning("tryLoadLibraryFailed: " + e.getMessage());
            return false;
        }
    }

    private static boolean isLoaded() {
        try {
            Version.version();
            LOGGER.info("isLoaded: true");
            return true;
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }

    private static void loadLib(String jniResourceDir, String jniResourceName) {
        LOGGER.info("start load jniResourceName: " + jniResourceName);
        final Integer BUFFER_SIZE = 8024;
        final String TMPDIR_PROPERTY = "java.io.tmpdir";
        try {
            InputStream in = NativeLibrary.class.getClassLoader().getResourceAsStream(jniResourceName);
            if (in == null || in.available() == 0) {
                LOGGER.severe(String.format("jni file: %s not exist.", jniResourceName));
                return;
            }
            String tmpPath = System.getProperty(TMPDIR_PROPERTY) + "/" + jniResourceDir;
            File fileOutDir = new File(tmpPath);
            if (!fileOutDir.exists()) {
                fileOutDir.mkdirs();
            }
            File fileOut = new File(tmpPath + jniResourceName.substring(jniResourceName.lastIndexOf("/") + 1));
            if (!fileOut.exists()) {
                fileOut.createNewFile();
            }

            OutputStream out = new FileOutputStream(fileOut);
            IOUtils.copy(in, out, BUFFER_SIZE);
            in.close();
            out.close();
            System.load(fileOut.getAbsolutePath());
            LOGGER.info(String.format("load file: %s success.", fileOut.getAbsolutePath()));
        } catch (IOException e) {
            throw new UnsatisfiedLinkError(String.format(
                    "Unable to extract native library into a temporary file (%s)", e.getMessage()));
        }
    }

    /**
     * get native lib dir, eg. temp/linux_x86_64
     * @param dir
     * @return
     */
    private static String makeResourceDir(String dir) {
        return dir + "/" + String.format("linux_%s/", architecture());
    }

    /**
     * get native lib file name, eg. com/mindspore/lite/linux_x86_64/libmindspore-lite.so
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
