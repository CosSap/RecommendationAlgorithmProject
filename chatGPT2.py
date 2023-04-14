import ssl
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def content_based_algorithm(game_title, platform, num_recommendations=10):
    # load games dataset
    all_games_df = pd.read_csv('Video Games Dataset.csv', encoding='latin1')

    # validation function must be used here
    validation(all_games=all_games_df, game_title=game_title, platform=platform)

    filtered_games_df = all_games_df
    filtered_games_df = filtered_games_df.reset_index(drop=False)
    filtered_games_df['description'] = filtered_games_df['description'].apply(preprocess_dataset)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(filtered_games_df['description'])

    cosine_sim_desc = cosine_similarity(tfidf_matrix)

    # select features
    features = ['Publisher', 'Year']

    # create feature vectors
    feature_vectors = pd.get_dummies(filtered_games_df[features])
    feature_vectors = feature_vectors.fillna(0)

    # compute pairwise similarity
    cosine_sim_rest = cosine_similarity(feature_vectors)

    # create feature vectors
    feature_vectors = pd.get_dummies(filtered_games_df[['Genre']])
    feature_vectors = feature_vectors.fillna(0)

    # compute pairwise similarity
    cosine_sim_genre = cosine_similarity(feature_vectors)

    return recommend_games(
        games_df=filtered_games_df,
        title=game_title,
        platform=platform,
        cosine_sim_descriptions=cosine_sim_desc,
        cosine_sim_year_and_publisher=cosine_sim_rest,
        cosine_sim_genre=cosine_sim_genre,
        num_recommendations=num_recommendations
    )


def downloads():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')
    nltk.download('stopwords')


def preprocess_dataset(column):
    # Convert to lowercase
    column = column.lower()

    # Remove punctuation
    column = column.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text into words
    tokens = word_tokenize(column)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stem words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join tokens back into text
    column = ' '.join(tokens)

    return column


def recommend_games(
        games_df,
        title,
        platform,
        cosine_sim_descriptions,
        cosine_sim_year_and_publisher,
        cosine_sim_genre,
        num_recommendations=10
):
    # Get the index of the given game title
    idx = games_df[(games_df['Name'] == title) & (games_df['Platform'] == platform)].index[0]

    # Get the cosine similarity scores for all games
    sim_scores_desc = list(enumerate(cosine_sim_descriptions[idx]))
    description_weight = 0.2

    # Get the cosine similarity scores for all games
    sim_scores_year_and_publishers = list(enumerate(cosine_sim_year_and_publisher[idx]))
    sim_scores_year_and_publishers_weight = 0.3

    sim_scores_genre = list(enumerate(cosine_sim_genre[idx]))
    sim_scores_genre_genre = 0.5

    sim_scores_temp = []
    for i in range(len(sim_scores_desc)):
        sim_scores_temp.append((i, sim_scores_desc[i][1] * description_weight +
                                sim_scores_year_and_publishers[i][1] * sim_scores_year_and_publishers_weight +
                                sim_scores_genre[i][1] * sim_scores_genre_genre,))

    sim_scores = sim_scores_temp

    # Sort the scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar games
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get the indices of the recommended games
    game_indices = [i[0] for i in sim_scores]

    # Removes the selected game
    games_df = games_df.drop(games_df[games_df['Name'] == title].index)

    for i in games_df:
        for y in games_df:
            if i.games_df['Name'] != y.games_df['Name']:
                y += 1
            else:
                games_df = games_df.games_df_filtered(['i','y'])


    # Return the recommended games
    return games_df.iloc[game_indices][['Name', 'Platform']]


def validation(all_games, game_title, platform):
    df_validation = all_games

    if game_title not in df_validation['Name'].values and platform not in df_validation['Platform'].values:
        raise ValueError("The game " + game_title + " and platform "+ platform + " do not exist in the database.")

    elif game_title not in df_validation['Name'].values:
        raise ValueError("The game " + game_title + " does not exist in the database.")

    elif platform not in df_validation['Platform'].values:
        raise ValueError("The platform " + platform + " does not exist in the database")

    else:
        print("The game " + game_title + " and platform "+ platform + " exist!")
        return


if __name__ == '__main__':
    # problems: Duplicated data, search accuracy,
    # print(recommend_games('FIFA 17', 'PS4'))

    results = content_based_algorithm(game_title='FIFA 16', platform='PS4', num_recommendations=10)
    print(results)

    # TODO: Inside our results we don't want duplicated games. Create a filter which removes the games that are duplicated.
    # TODO: The record that the user searches with has to be removed from the dataset
    # TODO: Transfer the chatGPT2.py file and all the necessary files into GitHub for extra security

