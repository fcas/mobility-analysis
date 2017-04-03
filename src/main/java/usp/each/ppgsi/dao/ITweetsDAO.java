package usp.each.ppgsi.dao;

import usp.each.ppgsi.model.Address;
import usp.each.ppgsi.model.TweetInfo;

import java.util.List;

/**
 * Created by felipealvesdias on 02/04/17.
 */
public interface ITweetsDAO {
    void saveTweetInfos(List<TweetInfo> tweetInfoList);

    void dropCollection();

    void closeMongo();

    void saveAddresses(List<Address> addresses);

    Address getAddress(String address);

    TweetInfo getTweet(long id);
}
