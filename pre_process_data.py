import pandas as pd
import re

raw_data_link = "ADD LINK HERE"
tweet_df = pd.read_csv(raw_data_link)
tweet_df['TweetAvgAnnotation'] = tweet_df['AverageAnnotation']
tweet_df.drop('AverageAnnotation', axis=1)

user_df = pd.read_csv('/content/drive/My Drive/cs109b-finalproject/user_information.csv')
user_df['UserAvgAnnotation'] = user_df['AverageAnnotation']
user_df.drop('AverageAnnotation', axis=1)

df = pd.merge(tweet_df, user_df, on='Username')
df = df.drop('AverageAnnotation_y', axis=1)
df = df.drop('AverageAnnotation_x', axis=1)

def extract_info(profile_info_str):
    if not isinstance(profile_info_str, str):
        return pd.Series({
          'UserDescription': None,
          'Followers': None,
          'Following': None,
          'TotalTweetCount': None,
          'FavoritesCount': None
        })

    info_dict = {}

    patterns = {
        'UserDescription': r'description: (.*?),',
        'Followers': r'followers: (\d+)',
        'Following': r'following: (\d+)',
        'TotalTweetCount': r'total tweet number: (\d+)',
        'FavoritesCount': r'favorites_count: (\d+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, profile_info_str)
        if match:
            info_dict[key] = match.group(1).strip()
        else:
            info_dict[key] = None

    return pd.Series(info_dict)

new_columns = df['ProfileInfo'].apply(extract_info)
df = pd.concat([df, new_columns], axis=1)
df = df.drop('ProfileInfo', axis=1)

def label_value(x):
    if x < -1:
        return 'Negative'
    elif x > 1:
        return 'Positive'
    else:
        return 'Neutral'

df['Sentiment'] = df['TweetAvgAnnotation'].apply(label_value)