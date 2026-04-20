"""Synthetic YouTube-style dataset for the Spotlight demo.

Each entry mimics a video with four human-authored annotation fields:

- ``title`` — the video title.
- ``subtitle`` — a short snippet of the spoken script (human speech).
- ``audio_caption`` — a short description of the non-speech soundscape (what the
  background audio *sounds like*, not what anyone says). This is the field the
  ``"meow"`` query in ``query.py`` targets.
- ``thumbnail_url`` — deterministic placeholder from picsum.photos, one per
  object_id so the Spotlight gallery looks stable across reruns.
- ``video_url`` — cycled from a small pool of public CC-BY sample MP4s so the
  Spotlight video lens has something to play without bundling media files.

The dataset is intentionally small (~60 rows across 6 categories) so the demo
runs end-to-end in under a minute.
"""

from __future__ import annotations

_SAMPLE_VIDEO_URLS = [
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
]


_RAW_ENTRIES: list[tuple[str, str, str, str]] = [
    # (category, title, subtitle, audio_caption)
    # ---- cats ----
    (
        "cats",
        "Kitten discovers a laser pointer",
        "Look at her go, she's so fast!",
        "rapid paw taps on wood and excited meowing",
    ),
    (
        "cats",
        "My cat refuses to share the couch",
        "Okay buddy, that's my spot.",
        "low rumbling purr and occasional short meow",
    ),
    (
        "cats",
        "Three cats singing at 3am",
        "Why do they do this every night?",
        "overlapping meowing and yowling cats",
    ),
    (
        "cats",
        "Cat vs. cucumber",
        "I did not expect that reaction.",
        "startled cat hissing and meowing loudly",
    ),
    (
        "cats",
        "Rescue kittens first day home",
        "They were hiding for hours.",
        "soft kitten mewing and quiet purring",
    ),
    (
        "cats",
        "Talkative Siamese answers every question",
        "You tell me all about it.",
        "long drawn-out meowing with vocal trills",
    ),
    (
        "cats",
        "Cat bathing in a sunbeam",
        "Don't move, don't move.",
        "gentle purring and distant birdsong through a window",
    ),
    (
        "cats",
        "Angry tabby demands breakfast",
        "It's not even six in the morning.",
        "insistent meowing and tail thumping",
    ),
    (
        "cats",
        "Two kittens wrestling",
        "They are absolutely not fighting, promise.",
        "high-pitched meowing and playful squeaks",
    ),
    (
        "cats",
        "Cat purrs into the microphone",
        "Pure ASMR, ten out of ten.",
        "continuous deep purring very close to microphone",
    ),
    # ---- dogs ----
    (
        "dogs",
        "Golden retriever meets a bubble machine",
        "He has no idea what's happening.",
        "excited barking and panting",
    ),
    (
        "dogs",
        "Husky argues about bath time",
        "You love the bath, you know you do.",
        "howling and grumbling vocalizations from a husky",
    ),
    (
        "dogs",
        "Puppy tries stairs for the first time",
        "You can do it, one step at a time.",
        "small whimpers and tiny paws on carpet",
    ),
    (
        "dogs",
        "Beagle chases the mailman",
        "I'm so sorry, I'm so sorry.",
        "frantic barking and jingling collar tags",
    ),
    (
        "dogs",
        "Dog parkour champion 2025",
        "That landing was incredible.",
        "fast running paws on pavement and quick panting",
    ),
    (
        "dogs",
        "Two dachshunds share one bed",
        "Nobody is moving, nobody wins.",
        "soft snoring and quiet dog breathing",
    ),
    (
        "dogs",
        "Border collie herds the toddler",
        "He thinks she's a sheep.",
        "short controlled barking and pattering paws",
    ),
    (
        "dogs",
        "My shepherd sings with the siren",
        "Every single time, without fail.",
        "long mournful dog howling overlaying a distant siren",
    ),
    (
        "dogs",
        "Dog reacts to squeaky toy",
        "He's obsessed with this one.",
        "high-pitched squeaking toy and muffled barking",
    ),
    (
        "dogs",
        "Training a rescue puppy week one",
        "Good girl, yes, good girl.",
        "clicker training sounds and soft encouraging whistles",
    ),
    # ---- music ----
    (
        "music",
        "Bedroom indie song I wrote last night",
        "I stayed up way too late finishing this.",
        "acoustic guitar strumming with soft vocals",
    ),
    (
        "music",
        "Guitar solo over a drum loop",
        "This riff has been stuck in my head.",
        "electric guitar solo with distortion and steady drum beat",
    ),
    (
        "music",
        "Piano cover of a movie theme",
        "Let me know what I should play next.",
        "solo piano playing a slow melodic theme",
    ),
    (
        "music",
        "Lo-fi beats to study to",
        "Perfect for a rainy afternoon.",
        "mellow lo-fi hip hop beat with vinyl crackle",
    ),
    (
        "music",
        "Drum kit practice session",
        "Working on paradiddles this week.",
        "rapid snare drum rolls and cymbal crashes",
    ),
    (
        "music",
        "Violin busker in the subway",
        "He played for hours, it was beautiful.",
        "solo violin playing a classical piece with subway ambience",
    ),
    (
        "music",
        "Synthwave track I produced",
        "All analog, no samples.",
        "retro synthesizer pads with electronic drum machine",
    ),
    (
        "music",
        "Choir rehearsal for spring concert",
        "Section leaders take it from bar twelve.",
        "layered choral singing in a reverberant hall",
    ),
    (
        "music",
        "Acoustic bass unboxing and jam",
        "First impressions of this new instrument.",
        "warm acoustic bass line with finger slides",
    ),
    (
        "music",
        "Electronic dance festival mainstage",
        "The crowd went absolutely wild here.",
        "thumping four-on-the-floor kick drum with synth build",
    ),
    # ---- cooking ----
    (
        "cooking",
        "Perfect steak in ten minutes",
        "Let the pan get smoking hot first.",
        "sizzling meat on a cast iron skillet",
    ),
    (
        "cooking",
        "Sourdough from starter to loaf",
        "Patience is the key ingredient here.",
        "quiet kitchen ambience and dough kneading",
    ),
    (
        "cooking",
        "Spicy ramen from scratch",
        "The broth simmered for eight hours.",
        "gentle bubbling broth and slurping noodles",
    ),
    (
        "cooking",
        "Chopping onions without tears",
        "The trick is a very sharp knife.",
        "rhythmic knife chopping on a wooden cutting board",
    ),
    (
        "cooking",
        "Grandma's Sunday pasta sauce",
        "She never measured anything, ever.",
        "tomato sauce simmering and wooden spoon stirring",
    ),
    (
        "cooking",
        "Ice cream with liquid nitrogen",
        "Don't try this without gloves.",
        "hissing nitrogen vapor and metal whisk on a bowl",
    ),
    (
        "cooking",
        "Knife skills for beginners",
        "Curl your fingers, keep the tip down.",
        "precise knife taps on a plastic cutting board",
    ),
    (
        "cooking",
        "Grilling over charcoal at the beach",
        "The smoke is doing the seasoning for us.",
        "crackling charcoal fire with ocean waves in the distance",
    ),
    (
        "cooking",
        "Baking a birthday cake with my kid",
        "More sprinkles, he insisted.",
        "electric hand mixer whirring and batter pouring",
    ),
    (
        "cooking",
        "Espresso machine deep clean",
        "You should do this every week.",
        "steam wand hissing and water running through a portafilter",
    ),
    # ---- gaming ----
    (
        "gaming",
        "Speedrunning an old platformer",
        "Frame perfect jump, let's go!",
        "retro chiptune music and rapid controller button presses",
    ),
    (
        "gaming",
        "Horror game first playthrough",
        "I am never playing this at night again.",
        "ominous low drone and sudden jump scare sting",
    ),
    (
        "gaming",
        "Co-op raid with my guild",
        "Healers, stay behind the pillar!",
        "overlapping voice chat and fast keyboard typing",
    ),
    (
        "gaming",
        "Strategy game one more turn syndrome",
        "It is four in the morning.",
        "quiet ambient map music and mouse clicks",
    ),
    (
        "gaming",
        "Racing sim on a wheel setup",
        "Into turn twelve, brake, brake, brake!",
        "engine revs and tire squeal through racing simulator speakers",
    ),
    (
        "gaming",
        "Battle royale final circle",
        "Two squads left, we can win this.",
        "gunfire and distant explosions over a game score",
    ),
    (
        "gaming",
        "Retro arcade beat em up",
        "Quarter after quarter, this was my childhood.",
        "eight-bit arcade music with coin insert jingles",
    ),
    (
        "gaming",
        "Puzzle game daily challenge",
        "I almost had it that time.",
        "gentle puzzle game chimes and soft success fanfare",
    ),
    (
        "gaming",
        "Rhythm game expert chart",
        "Missed one note, one single note.",
        "synchronized tap sounds over an upbeat electronic track",
    ),
    (
        "gaming",
        "Indie adventure game finale",
        "That ending got me, I won't lie.",
        "emotional orchestral swell and subtle keyboard taps",
    ),
    # ---- nature ----
    (
        "nature",
        "Dawn chorus in an old forest",
        "I hiked in at four to catch this.",
        "layered birdsong at dawn in a dense forest",
    ),
    (
        "nature",
        "Waterfall from the top of the ridge",
        "The spray reaches all the way up here.",
        "roaring waterfall with wind through pine trees",
    ),
    (
        "nature",
        "Whale watching off the coast",
        "Two breaches in one minute!",
        "distant whale calls and small boat engine idling",
    ),
    (
        "nature",
        "Thunderstorm rolling across the plains",
        "No better sound for sleeping, honestly.",
        "rumbling thunder and steady rain on a metal roof",
    ),
    (
        "nature",
        "Tide pools at low tide",
        "Every crevice is full of life.",
        "gentle ocean waves and clicking crab shells",
    ),
    (
        "nature",
        "Frogs after a summer rain",
        "The whole pond wakes up at once.",
        "overlapping frog croaks and chirping crickets",
    ),
    (
        "nature",
        "Alpine meadow in July",
        "Wildflowers everywhere you look.",
        "light wind through grass and buzzing bees",
    ),
    (
        "nature",
        "Desert night under the stars",
        "No light pollution for a hundred miles.",
        "very quiet desert ambience with faint coyote howls",
    ),
    (
        "nature",
        "Autumn leaves in the northern woods",
        "Every step sounds like cereal.",
        "crunching dry leaves and cold wind through branches",
    ),
    (
        "nature",
        "River rapids at snowmelt",
        "The water is freezing, do not fall in.",
        "rushing river water over rocks and distant birdsong",
    ),
]


def load_videos() -> list[dict]:
    """Return the full list of sample video entries.

    Each entry is a plain dict, ready to pass to `Collection.add_many` after the
    caller attaches an embedding under the ``vectors`` key.
    """
    entries: list[dict] = []
    for idx, (category, title, subtitle, audio_caption) in enumerate(_RAW_ENTRIES):
        object_id = f"vid_{idx:03d}"
        entries.append(
            {
                "object_id": object_id,
                "title": title,
                "subtitle": subtitle,
                "audio_caption": audio_caption,
                "category": category,
                "thumbnail_url": f"https://picsum.photos/seed/{object_id}/320/180",
                "video_url": _SAMPLE_VIDEO_URLS[idx % len(_SAMPLE_VIDEO_URLS)],
            }
        )
    return entries


if __name__ == "__main__":
    videos = load_videos()
    print(f"{len(videos)} sample videos")
    from collections import Counter

    print(Counter(v["category"] for v in videos))
