import math
import os
import nltk
import numpy
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

nltk.download('punkt')


def read_files_to_dictionaries(directory_path):
    """
    This method will iterate on all the files in the input sub-directories and for each song calculates the term
    frequencies
    @param directory_path: directory path where all the directories of different genre are located
    @return: dictionary with song name as the key and the value as a dictionary (A). (A) is a dictionary with key
    representing the token and value being the frequency in the document. The second return type will give you genre for
    each song.
    """
    dic_song_term_frequency = {}
    dic_song_genre = {}
    for genre in os.listdir(directory_path):
        path = directory_path + genre
        for file in os.listdir(path):
            temp_dic = {}
            with open(path + "/" + file, 'r') as rfile:
                for line in rfile:
                    current_line = line.strip()
                    current_line = current_line.lower()
                    current_line = re.sub(r'[^\w\s]', '', current_line)
                    token_list = word_tokenize(current_line)
                    for token in token_list:
                        # Update frequency for existing term or add a new term with frequency 1
                        temp_dic[token] = temp_dic.get(token, 0) + 1
            song_name = file.split(".")[0]
            dic_song_term_frequency[song_name] = temp_dic
            dic_song_genre[song_name] = genre
    return dic_song_term_frequency, dic_song_genre


def get_TF_values(dic_song_term_frequency):
    """
    This method takes in token frequency per song as the input and returns TF per token/song
    @param dic_song_term_frequency: song name as key and dictionary A as value. In A, keys are the tokens with their
    frequencies as the values
    @return: Dictionary with song names as keys, and TF-Values as values. These values are also a dictionary of token as
    the keys and their TFs as values
    """
    dic_tf_per_song = {}
    for song in dic_song_term_frequency:
        ############################
        # Put your code here and then remove 'pass'
        # YOUR CODE
        ############################
        pass

    return dic_tf_per_song


def get_IDF_values(dic_song_term_frequency):
    """
    This method calculates the IDF values for each token
    @param dic_song_term_frequency: song name as key and dictionary A as value. In A, keys are the tokens with their
    frequencies as the values
    @return: Dictionary with tokens as the keys and IDF values as values
    """
    dic_idf_values = {}
    num_songs = len(dic_song_term_frequency.keys())

    for song in dic_song_term_frequency:
        for word in dic_song_term_frequency[song]:
            # Update df for existing word or add a new word with df 1
            dic_idf_values[word] = dic_idf_values.get(word, 0) + 1

    for word in dic_idf_values:
        # The values of dic_idf_values are currently df, change them to idf
        dic_idf_values[word] = math.log(num_songs / dic_idf_values[word])
    return dic_idf_values


def song_to_vector(dic_tf_per_song, dic_idf):
    """
    This method will extract vector representation (based on TF-IDF) for each song
    @param dic_tf_per_song: TF dictionary with song name as key and dictionary of (token:TF) as value
    @param dic_idf: IDF dictionary with token as key and its IDF as value
    @return: dictionary of (song name, vector)
    """
    song_to_vec = {}
    for song in dic_tf_per_song:
        ############################
        # Put your code here and then remove 'pass'
        # Sample code to get numpy array from a python list: numpy.array(lst)
        # Sample code to sort python dictionary based on keys dict(sorted(my_dic.items()))
        # YOUR CODE
        ############################
        pass
    return song_to_vec


def cosine_sim(numpy_vec1, numpy_vec2):
    """
    Using sklearn library, this method calculates the cosine similarity between two numpy vectors
    @param numpy_vec1: vector 1
    @param numpy_vec2: vector 2
    @return: cosine similarity
    """
    return cosine_similarity(numpy_vec1, numpy_vec2)


def test_cosine(dic_song_vectors):
    print(cosine_sim(dic_song_vectors["Till I Collapse"].reshape(-1, 1), dic_song_vectors["Rap God"].reshape(-1, 1)))
    print(
        cosine_sim(dic_song_vectors["Till I Collapse"].reshape(-1, 1), dic_song_vectors["Billie Jean"].reshape(-1, 1)))


def test_tsne_plot(dic_song_vectors, dic_song_genre):
    lst_song_names = []
    lst_data = []
    colors = []
    tsne = TSNE(n_components=2)
    for song in list(dic_song_vectors.keys()):
        lst_song_names.append(song)
        lst_data.append(dic_song_vectors[song])
        color = get_color(dic_song_genre[song])
        colors.append(color)
    data = numpy.array(lst_data)
    vectors = tsne.fit_transform(data)
    ax = plt.axes()
    ax.set_facecolor("black")
    for i in range(len(vectors)):
        plt.scatter(vectors[i][0], vectors[i][1], color=colors[i])

    # Adding song names; this can make the plot messy; you can choose a fewer songs and uncomment this code
    # for i, txt in enumerate(lst_song_names):
    #     plt.annotate(txt, (vectors[i][0], vectors[i][1]))
    plt.show()


def get_color(genre):
    """
    Assigns a color to each genre
    @param genre: song genre
    @return: genre's color
    """
    if genre == "Blues":
        color = 'blue'
    elif genre == "Country":
        color = 'red'
    elif genre == "Metal":
        color = 'gray'
    elif genre == "Pop":
        color = 'yellow'
    elif genre == "Rap":
        color = 'green'
    elif genre == "Rock":
        color = 'purple'
    return color


def main():
    # Path to the root Lyrics files
    path_to_root_dir = r"C:\Users\Gigabyte\Desktop\Genres"
    dic_song_dic_term_count, dic_song_genre = read_files_to_dictionaries(path_to_root_dir + "/")
    # dic_song_dic_term_frequency = get_TF_values(dic_song_dic_term_count)
    dic_term_idfs = get_IDF_values(dic_song_dic_term_count)
    # dic_song_vectors = song_to_vector(dic_song_dic_term_frequency, dic_term_idfs)
    # test_cosine(dic_song_vectors)
    # test_tsne_plot(dic_song_vectors, dic_song_genre)


if __name__ == '__main__':
    main()
