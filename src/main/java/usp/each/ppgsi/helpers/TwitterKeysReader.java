package usp.each.ppgsi.helpers;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Created by felipealvesdias on 02/04/17.
 */
public class TwitterKeysReader {
    private static final String configurationFileName = "config.properties";
    private static TwitterKeysReader instance;

    private TwitterKeysReader() {
    }

    public static Properties getTwitterKeys() {
        InputStream _input = TwitterKeysReader.class.getClassLoader().getResourceAsStream(configurationFileName);

        if (instance == null) {
            instance = new TwitterKeysReader();
        }

        Properties properties = new Properties();

        try {
            if (_input == null) {
                System.out.println("Sorry, unable to find " + configurationFileName);
                return null;
            }
            properties.load(_input);
            properties.setProperty("accessToken", properties.getProperty("accessToken"));
            properties.setProperty("accessTokenSecret", properties.getProperty("accessTokenSecret"));
            properties.setProperty("consumerKey", properties.getProperty("consumerKey"));
            properties.setProperty("consumerSecret", properties.getProperty("consumerSecret"));
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            if (_input != null) {
                try {
                    _input.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return properties;
    }
}
