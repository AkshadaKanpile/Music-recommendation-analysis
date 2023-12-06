from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import pickle

app = Flask(__name__)
model = joblib.load("spotify.pkl")
# Load your Spotify data and pre-processing here
# For the recommendation function, you can use the modified code provided earlier

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    song_name = request.form["song_name"]
    song_year = int(request.form["song_year"])

    # Call your recommendation function with the input song details
    def recommend_songs(song_list, spotify_data, model, n_songs=10):
        metadata_cols = ['name', 'year', 'artists']
        song_dict = flatten_dict_list(song_list)
    
        # Prepare the input data for the model (you need to adjust this based on your model's requirements)
        input_data = preprocess_input_data(song_list, spotify_data)
    
        # Use your pre-trained model to make recommendations
        recommended_songs_indices = model.predict(input_data)
    
        # Get the recommended songs from the Spotify data
        rec_songs = spotify_data.iloc[recommended_songs_indices]
    
        # Filter out songs that are already in the input list
        rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
        # Return the recommended songs
        return rec_songs[metadata_cols].to_dict(orient='records')

    # Call the recommend_songs function
    recommended_songs = recommend_songs(song_list, spotify_data, model)
    
    # Return the recommendations as JSON
    return jsonify(recommended_songs)

if __name__ == "__main__":
    app.run(host ="0.0.0.0",port=8080)
