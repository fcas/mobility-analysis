package usp.each.ppgsi.tweets;

import org.slf4j.LoggerFactory;
import twitter4j.*;
import twitter4j.auth.AccessToken;

import usp.each.ppgsi.dao.ITweetsDAO;
import usp.each.ppgsi.dao.TweetsDAO;
import usp.each.ppgsi.helpers.AccountsReader;
import usp.each.ppgsi.helpers.TwitterKeysReader;
import usp.each.ppgsi.model.Address;
import usp.each.ppgsi.model.TweetInfo;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;


/**
 * Created by felipealvesdias on 02/04/17.
 */
public class TweetService {
    private static final String dbBaseName = "tweets_";
    private static ITweetsDAO tweetsDao;
    private static Twitter twitter;
    private static AccessToken token;
    private static List<TweetInfo> tweetInfoList;
    private static List<Address> addresses;

    private static final org.slf4j.Logger logger = LoggerFactory.getLogger(TweetService.class);


    public TweetService() {
        authentication();
    }

    private static void authentication() {
        Properties properties = TwitterKeysReader.getTwitterKeys();

        System.setProperty("twitter4j.oauth.consumerKey", (String) properties.get("consumerKey"));
        System.setProperty("twitter4j.oauth.consumerSecret", (String) properties.get("consumerSecret"));
        System.setProperty("twitter4j.oauth.accessToken", (String) properties.get("accessToken"));
        System.setProperty("twitter4j.oauth.accessTokenSecret", (String) properties.get("accessTokenSecret"));
        token = new AccessToken((String) properties.get("accessToken"), (String) properties.get("accessTokenSecret"));
        twitter = new TwitterFactory().getInstance();
        try {
            twitter.setOAuthConsumer((String) properties.get("consumerKey"), (String) properties.get("consumerSecret"));
            twitter.setOAuthAccessToken(token);
        } catch (Exception ignored) {
        }
    }

    public static void processTweets() {
        tweetsDao = new TweetsDAO();
        List<String> accounts = new ArrayList<>(AccountsReader.getAccounts().values());

        for (int i = 0; i < accounts.size(); i++) {
            String account = accounts.get(i);
            try {
                processUserTimelines(account);
            } catch (TwitterException e) {
                e.printStackTrace();
            }
            tweetsDao.saveAddresses(addresses, dbBaseName + "locations");
        }
        tweetsDao.closeMongo();
    }

    private static void processUserTimelines(String username) throws TwitterException {
        tweetInfoList = new ArrayList<>();
        addresses = new ArrayList<>();

        int pageCounter = 1;
        int pageLimit = 200;

        ResponseList<Status> userTimeLine = null;
        RateLimitStatus rateLimitStatus = twitter.getRateLimitStatus().get("/statuses/user_timeline");
        int remaining = rateLimitStatus.getRemaining();
        do {
            if (remaining == 0) {
                try {
                    Thread.sleep(900000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                userTimeLine = twitter.getUserTimeline(
                        username, new Paging(pageCounter, pageLimit));
                if (userTimeLine.size() > 0) {
                    processUserTweets(userTimeLine, username);
                    pageCounter++;
                }
            }
            remaining--;
        } while (remaining > 0 & pageCounter != 17 && userTimeLine != null && !userTimeLine.isEmpty());

        tweetsDao.saveTweetInfos(tweetInfoList, dbBaseName + username);
    }

    private static void processUserTweets(ResponseList<Status> tweets, String username) {
        for (Status status : tweets) {
            long statusId = status.getId();
            TweetInfo tweetInfo = tweetsDao.getTweet(statusId, dbBaseName + username);
            if (tweetInfo == null) {
                String text = status.getText();
//                if (text.contains("#ZC") || text.contains("#ZN") || text.contains("#ZO") || text.contains("#ZS") || text.contains("#ZL")) {
                    String found_address = TweetProcessor.getAddress(text);
                    LocalDateTime localDateTime = status.getCreatedAt().toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime();
                    String dateTime = localDateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                    tweetInfo = new TweetInfo();
                    tweetInfo.setTweetId(statusId);
                    tweetInfo.setDateTime(dateTime);
                    tweetInfo.setTweetText(text);
                    if (!found_address.isEmpty()) {
                        tweetInfo = processAddress(tweetInfo, found_address);
                    }
                    tweetInfoList.add(tweetInfo);
//                }
            }
        }
    }

    private static TweetInfo processAddress(TweetInfo tweetInfo, String found_address) {
        try {
            Address address = tweetsDao.getAddress(found_address, dbBaseName + "locations");
            if (address == null) {
                org.json.JSONObject address_info = GoogleMapsService.getLatLong(found_address.replace(" ", "+"));
                if (address_info != null) {
                    tweetInfo = MapJsonToTweetInfo(tweetInfo, address_info);
                }
            } else {
                tweetInfo.setAddress(address.getAddress());
                tweetInfo.setLat(address.getLat());
                tweetInfo.setLng(address.getLng());
            }
        } catch (Exception e) {
            logger.error("Error getting address from " + found_address, e);
        }
        return tweetInfo;
    }

    private static TweetInfo MapJsonToTweetInfo(TweetInfo tweetInfo, org.json.JSONObject jsonObject) {

        String formatted_address = jsonObject.getString("formatted_address");
        org.json.JSONObject geometry = jsonObject.getJSONObject("geometry");
        org.json.JSONObject location = geometry.getJSONObject("location");
        String id = jsonObject.getString("place_id");
        String lat = Double.toString(location.getDouble("lat"));
        String lng = Double.toString(location.getDouble("lng"));

        tweetInfo.setAddress(formatted_address);
        tweetInfo.setLat(lat);
        tweetInfo.setLng(lng);

        Address address = new Address(id, formatted_address, lat, lng);
        addresses.add(address);

        return tweetInfo;
    }

    public static void main(String[] args) {
        authentication();
        processTweets();
    }
}
