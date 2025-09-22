import generator

prompt_weeknd = "Write a new song in the style of The Weeknd about heartbreak at night while he was high on ecstasy and oxycontin. He mentions about prioritising his career over his love which led to the heartbreak. YOur output should only consist of the lyrics and nothing else. NO [verse1] or no [chorus] or not even pre chorus or no instruments or anything like that. Only and only the lyrics, no instructions whatsoever either."
seed_lyrics_future = """
I'm drifting under neon lights,  
With echoes of your touch last night,  
My heart's a beat behind the sound,  
But you're not here, I'm breaking down."""

prompt_future = f"""Continue this song in the same style and emotion:\n\n{seed_lyrics_future}\nYOur output should only consist of the lyrics and nothing else. NO [verse1] or [chorus] or even pre chorus or instruments or anything like that. Only and only the lyrics, no instructions whatsoever either."""

seed_lyrics_hoz = """
I found your voice beneath cathedral rain,  
Like hymns carved deep in sacred stone,  
Your touch still lingers in the flame,  
A ghost that sings when I’m alone."""

prompt_hoz = f"""You're a poetic lyricist like Hozier, known for soulful and haunting metaphors, blending love and spirituality. Continue this song in the same tone and rhythm:\n\n{seed_lyrics_hoz}\nYOur output should only consist of the lyrics and nothing else. NO [verse1] or [chorus] or even pre chorus or instruments or anything like that. Only and only the lyrics, no instructions whatsoever either."""

seed_lyrics_eminem = """
Back when I was thirteen, had rage in my veins,  
Mom workin' doubles just to handle the pain,  
No heat in the house, just beats in my brain,  
Used rap as my shelter, escaped in the rain."""

prompt_eminem = f"""You're a rapper like Eminem, telling raw, emotional stories with tight rhyme schemes and fast flow. Continue these lyrics with intensity and rhythm:\n\n{seed_lyrics_eminem}\n"""

predicted_artist = generator.main(prompt_eminem)