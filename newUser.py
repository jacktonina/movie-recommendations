import csv
import pandas as pd
import re
import time


def getInput():
    movies = pd.read_csv('data/movies.csv')
    movies['title'] = movies['title'].str[:-7]
    user = input("Enter UserID: ")
    regex = re.compile('^[0-9]{6,6}$')
    if regex.match(str(user)):
        movie = input("Enter name of movie: ")
        movie_id = movies.loc[movies['title'] == movie].iloc[0][0]
        if movie_id>0:
            rating = input("Enter your rating (1-5 scale, one decimal allowed): ")
            rating_regex = re.compile('^([0-5](\.\d)?)$')
            if rating_regex.match(str(rating)):
                print(f'Thank you for rating {movie}! Your rating has been entered.')
            elif not rating_regex.match(str(rating)):
                raise Exception("Rating not accepted. Please make sure your rating is either an interger "
                                "or has only one decimal place, and is between 0 and 5.")
        elif type(movie_id)!=int:
            raise Exception("Movie not found. Please make sure the movie is from 1995 or later and try again.")
    elif len(user) != 6:
        raise Exception("UserID must be six consecutive digits. Please try again.")
    to_write = [user, movie_id, rating]
    print(to_write)
    return to_write

def writeRow(entry):
    userID = entry[0]
    movieID = entry[1]
    rate = entry[2]
    with open('data/ratings.csv', 'a', newline='') as file:
        fieldnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({'user_id': userID, 'movie_id': movieID, 'rating': rate, 'timestamp': time.time()})

def main():
    add_entry = getInput()
    writeRow(entry=add_entry)

if __name__ == "__main__":
    main()
