package com.mindspore.flclient.common;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.Date;
import java.util.logging.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Generate Logger for FL, define the log format in the logging.properties file
 *
 * @author : zhangzhaoju
 * @since : 2022/4/14
 */
public class FLLoggerGenerater {
    private static final ConsoleHandler handler = new ConsoleHandler();
    private static final FLFormatter formatter = new FLFormatter();
    private static Level level = Level.INFO;
    /**
     * Log format define Class, custom the log format for FL
     *
     * @author : zhangzhaoju
     * @since : 2022/4/14
     */
    public static class FLFormatter extends Formatter {
        private final String format = "%1$tb %1$td, %1$tY %1$tl:%1$tM:%1$tS %1$Tp %2$s%n%4$s: <FLClient> %5$s%6$s%n";
        private final MsgAnonymous anonymous = new MsgAnonymous();
        private static boolean anonymousFlg = true;
        private final Date date = new Date();

        public FLFormatter() {
        }

        public static void setAnonymousFlg(boolean anonymousFlg) {
            FLFormatter.anonymousFlg = anonymousFlg;
        }

        public String format(LogRecord record) {
            date.setTime(record.getMillis());
            String source = record.getLoggerName();
            String message = this.formatMessage(record);
            String anoyedMsg = anonymousFlg ? anonymous.doAnonymous(message) : message;
            String throwable = "";
            if (record.getThrown() != null) {
                StringWriter strWrite = new StringWriter();
                PrintWriter printWrite = new PrintWriter(strWrite);
                printWrite.println();
                record.getThrown().printStackTrace(printWrite);
                printWrite.close();
                throwable = strWrite.toString();
            }
            return String.format(format, date, source, record.getLoggerName(), record.getLevel(), anoyedMsg, throwable);
        }
    }

    public static void setLevel(Level level) {
        FLLoggerGenerater.level = level;
    }

    /**
     * Define logger format for FL only
     *
     * @param clsName
     * @return
     */
    public static Logger getModelLogger(String clsName) {
        Logger logger = Logger.getLogger(clsName);
        logger.setUseParentHandlers(false);
        handler.setLevel(level);
        // Enable Anonymous while level is bigger than FINE
        FLFormatter.setAnonymousFlg(level.intValue() > Level.FINE.intValue());
        handler.setFormatter(formatter);
        logger.addHandler(handler);
        return logger;
    }
}
