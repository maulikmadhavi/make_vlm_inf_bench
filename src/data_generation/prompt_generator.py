import random

IMAGE_PROMPTS = [
    "Describe this image in detail.",
    "What objects can you see in this image?",
    "Summarize what is happening in this scene.",
    "What is the main subject of this image?",
    "Describe the colors, textures, and composition of this image.",
    "What time of day does this image appear to be taken?",
    "Are there any people in this image? If so, describe them.",
    "What is the setting or location shown in this image?",
    "What emotions or mood does this image convey?",
    "List all the objects you can identify in this image.",
]

VIDEO_PROMPTS = [
    "Describe in detail what happens in this video.",
    "What is the main activity or event shown in this video?",
    "Summarize the sequence of events in this video.",
    "What objects and people appear in this video?",
    "Describe the setting and environment shown in this video.",
    "What actions are being performed in this video?",
    "How would you describe the motion and movement in this video?",
    "What is the overall theme or subject of this video?",
    "Describe any notable events or changes that occur in this video.",
    "What story does this video tell?",
]


def get_image_prompt() -> str:
    return random.choice(IMAGE_PROMPTS)


def get_video_prompt() -> str:
    return random.choice(VIDEO_PROMPTS)
