import lyricsgenius
import os

save_path = "data"

GENIUS_API_TOKEN = os.environ.get("GENIUS_KEY")
if not GENIUS_API_TOKEN:
    raise RuntimeError("Set GENIUS_API_TOKEN environment variable.")

genius = lyricsgenius.Genius(GENIUS_API_TOKEN, skip_non_songs=True,
                             remove_section_headers=True, excluded_terms=["(Remix)", "(Live)"],
                             timeout=20, retries=3)

def scrape_lyrics(artist_name, max_songs, save_path):
    artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
    if not artist:
        print(f"Could not find artist: {artist_name}")
        return

    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"{artist_name.lower().replace(' ', '_')}_lyrics.txt")

    with open(filename, "w", encoding="utf-8") as f:
        for song in artist.songs:
            f.write(f"{song.title}\n")
            f.write(song.lyrics.replace("\n\n", "\n"))
            f.write("\n\n===\n\n")

    print(f"Saved {len(artist.songs)} songs for {artist.name} to {filename}")



artists = [
    "The Weeknd","Eminem","Future","Hozier","Taylor Swift","Billie Eilish",
    "Ed Sheeran","Lil Baby","Drake","Travis Scott","SZA","Post Malone","Ariana Grande",
]

for name in artists:
    scrape_lyrics(name, max_songs=100, save_path="lyrics")  # change fmt to 'txt' if you prefer
