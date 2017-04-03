package usp.each.ppgsi.model;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlRootElement;
import java.io.Serializable;

/**
 * Created by felipealvesdias on 02/04/17.
 */
@XmlRootElement(name = "TweetInfo")
@XmlAccessorType(XmlAccessType.FIELD)
public class TweetInfo implements Serializable {
    private Long _id;
    private String text;
    private String dateTime;
    private String lat;
    private String lng;
    private String address;

    public TweetInfo() {
    }

    public Long getTweetId() {
        return _id;
    }

    public void setTweetId(long id) {
        this._id = id;
    }

    public String getTweetText() {
        return text;
    }

    public void setTweetText(String text) {
        this.text = text;
    }

    public String getDateTime() {
        return dateTime;
    }

    public void setDateTime(String dateTime) {
        this.dateTime = dateTime;
    }

    public String getLat() {
        return lat;
    }

    public void setLat(String lat) {
        this.lat = lat;
    }

    public String getLng() {
        return lng;
    }

    public void setLng(String lng) {
        this.lng = lng;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }
}
