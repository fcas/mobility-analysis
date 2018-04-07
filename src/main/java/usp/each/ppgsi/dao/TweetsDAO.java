package usp.each.ppgsi.dao;

import com.mongodb.DB;
import com.mongodb.MongoClient;
import org.jongo.Jongo;
import usp.each.ppgsi.model.Address;
import usp.each.ppgsi.model.TweetInfo;

import java.util.List;

/**
 * Created by felipealvesdias on 02/04/17.
 */
public class TweetsDAO implements ITweetsDAO {
    private DB database;
    private Jongo jongo;
    private MongoClient mongoClient;

    public TweetsDAO()
    {
        mongoClient = new MongoClient();
        database = mongoClient.getDB("tweets_db5");
        jongo = new Jongo(database);
    }

    public void saveTweetInfos(List<TweetInfo> tweetInfoList, String collectionName)
    {
        for (TweetInfo tweetInfo : tweetInfoList)
        {
            jongo.getCollection(collectionName).save(tweetInfo);
        }
    }

    public void closeMongo()
    {
        mongoClient.close();
    }

    public void saveAddresses(List<Address> addresses, String collectionName) {
        for (Address address : addresses)
        {
            jongo.getCollection(collectionName).save(address);
        }
    }

    public Address getAddress(String address, String collectionName)
    {
        return jongo.getCollection(collectionName).
                findOne("{address: {$regex : \".*" + address +".*\"}}").
                as(Address.class);
    }

    public TweetInfo getTweet(long id, String collectionName)
    {
        return jongo.getCollection(collectionName).
                findOne("{_id:" + id + "}").
                as(TweetInfo.class);
    }
}
