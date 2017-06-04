package usp.each.ppgsi.dao;

import usp.each.ppgsi.model.Address;
import usp.each.ppgsi.model.TweetInfo;

import java.util.List;

/**
 * Created by felipealvesdias on 02/04/17.
 */
public interface ITweetsDAO {
    void saveTweetInfos(List<TweetInfo> tweetInfoList, String collectionName);

    void closeMongo();

    void saveAddresses(List<Address> addresses, String collectionName);

    Address getAddress(String address, String collectionName);

    TweetInfo getTweet(long id, String collectionName);
}
