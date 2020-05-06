# Use python to pull the data
import pandas as pd
import spotipy
sp = spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials
cid ="ENTER YOUR CLIENT ID HERE"
secret = "ENTER YOUR SECRET ID HERE"
# This assumes you've set your credentials as environmental variables (in python run, os.environ["Client ID"] = "ENTER YOUR CLIENT ID AS A STR HERE")
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False


## Spotify only lets you pull 100 songs at a time, these loops reiterate until all track info has been collected
def show_tracks(results,uriArray):
    for i, item in enumerate(results['items']):
        track=item['track']
        uriArray.append(track['id'])


def get_playlist_tracks(username,playlist_id):
    trackId = []
    results = sp.user_playlist(username,playlist_id)
    tracks=results["tracks"]
    show_tracks(tracks, trackId)
    while tracks['next']:
        tracks=sp.next(tracks)
        show_tracks(tracks, trackId)
    return trackId



goodPlaylist = get_playlist_tracks("ENTER USER NAME HERE", "ENTER PLAYLIST ID HERE")
badPlaylist = get_playlist_tracks("ENTER USER NAME HERE", "ENTER PLAYLIST ID HERE")


## The audio features limits pulls to 100 so I pulled 100 individually each time then concatenated them. I'm certain a for
# loop could be written to do this automatically, my dataset was just small enough that this was faster for me to get working.
goodPlaylist1=sp.audio_features(goodPlaylist[0:99])
goodPlaylist2=sp.audio_features(goodPlaylist[99:199])
goodPlaylist3=sp.audio_features(goodPlaylist[199:299])
goodPlaylist4=sp.audio_features(goodPlaylist[299:399])
goodPlaylist5=sp.audio_features(goodPlaylist[399:])

## Have to convert the split list data to dataframes before you can concatenate them
goodPlaylist1 = pd.DataFrame(goodPlaylist1)
goodPlaylist2 = pd.DataFrame(goodPlaylist2)
goodPlaylist3 = pd.DataFrame(goodPlaylist3)
goodPlaylist4 = pd.DataFrame(goodPlaylist4)
goodPlaylist5 = pd.DataFrame(goodPlaylist5)

goodPlaylist = pd.concat([goodPlaylist1, goodPlaylist2, goodPlaylist3, goodPlaylist4, goodPlaylist5])
goodFeatures= goodPlaylist.to_csv("goodFeatures.csv")



badPlaylist1 =sp.audio_features(badPlaylist[0:99])
badPlaylist2 =sp.audio_features(badPlaylist[99:199])
badPlaylist3 =sp.audio_features(badPlaylist[199:299])
badPlaylist4 =sp.audio_features(badPlaylist[299:399])
badPlaylist5 =sp.audio_features(badPlaylist[399:])

badPlaylist1 = pd.DataFrame(badPlaylist1)
badPlaylist2 = pd.DataFrame(badPlaylist2)
badPlaylist3 = pd.DataFrame(badPlaylist3)
badPlaylist4 = pd.DataFrame(badPlaylist4)
badPlaylist5 = pd.DataFrame(badPlaylist5)

badPlaylist = pd.concat([badPlaylist1, badPlaylist2, badPlaylist3, badPlaylist4, badPlaylist5])
badFeatures= badPlaylist.to_csv("badFeatures.csv")



# Add column called "Like" to both dataframes. Set good playlist to 1, bad to 0. Then merge two dataframes into 1
goodPlaylist['Like']=1
badPlaylist['Like']=0

featuresData = pd.concat([goodPlaylist, badPlaylist])
featuresData.to_csv("featuresData.csv")