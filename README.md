# Social-Media-Poster
---------
To create Instagram images from text files stored in a folder, where each text file corresponds to a unique image, and then post them to Instagram, you can follow these steps:

    Load Text Files: Read the content of each text file.
    Generate Images: Combine the text from each file with a random cartoon image.
    Post to Instagram: Use the Instabot library (or another suitable library) to upload the generated images to Instagram.

Steps Breakdown:

    Prepare Random Cartoon Images: To create random cartoon images, you can either use a set of pre-downloaded cartoon images or integrate an API that generates cartoon-style images (like DeepAI's CartoonGAN API).
    Text to Image: Combine the loaded text with a random cartoon image, and overlay the text on top of the image using Pillow.
    Post to Instagram: Use the Instabot library to upload the generated images to Instagram.

Code Example:

import os
import random
from PIL import Image, ImageDraw, ImageFont
import requests
from instabot import Bot
from datetime import datetime

# Function to load text content from file
def load_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to generate image with text overlay
def generate_image_with_text(text, base_image_path="random_cartoon_image.jpg"):
    # Open a random cartoon image
    img = Image.open(base_image_path)
    
    # Set up drawing context
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    # Determine where to place the text on the image
    width, height = img.size
    text_width, text_height = draw.textsize(text, font)
    position = ((width - text_width) // 2, height - text_height - 20)  # Center the text at the bottom
    
    # Add text to image
    draw.text(position, text, font=font, fill="white")
    
    # Save the generated image
    image_name = f"generated_images/{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    img.save(image_name)
    
    return image_name

# Function to upload generated image to Instagram
def upload_to_instagram(image_path, caption):
    bot = Bot()
    bot.login(username="YOUR_INSTAGRAM_USERNAME", password="YOUR_INSTAGRAM_PASSWORD")
    bot.upload_photo(image_path, caption=caption)

# Function to generate and post content from text files
def generate_and_post_from_text_files():
    # Define the folder containing text files
    text_folder = "texts"
    
    # List all text files in the folder
    text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]
    
    for text_file in text_files:
        # Load the content of the text file
        text_path = os.path.join(text_folder, text_file)
        text_content = load_text_from_file(text_path)
        
        # Select a random cartoon image (you can customize this logic)
        random_cartoon_image = "cartoons/" + random.choice(os.listdir("cartoons"))
        
        # Generate the image with the text overlay
        image_path = generate_image_with_text(text_content, random_cartoon_image)
        
        # Upload to Instagram with the text content as caption
        upload_to_instagram(image_path, text_content)
        
        # Optionally, delete the image after posting to save space
        os.remove(image_path)

# Function to create folders if they don't exist
def create_folders():
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("cartoons", exist_ok=True)  # Folder with cartoon images
    os.makedirs("texts", exist_ok=True)  # Folder with text files

if __name__ == "__main__":
    create_folders()
    generate_and_post_from_text_files()

Explanation:

    Text Loading:
        The function load_text_from_file reads content from each .txt file stored in the texts folder.

    Image Generation:
        The function generate_image_with_text overlays the loaded text on a randomly selected cartoon image. You can place your cartoon images in the cartoons folder.
        It uses Pillow to create a new image with the text on top. You can adjust the text position, font, and size as necessary.

    Instagram Upload:
        The function upload_to_instagram uploads the generated image to Instagram with the text content as the caption.
        It uses the Instabot library to interact with Instagram. You'll need to install instabot and configure your Instagram credentials.

    Folder Structure:
        cartoons/: Folder containing random cartoon images to use as the base for each post.
        texts/: Folder containing .txt files, each with content that will be added to an image.
        generated_images/: Folder where the generated images are stored before being uploaded to Instagram.

    Run and Schedule:
        The generate_and_post_from_text_files function processes each text file, creates an image, and uploads it to Instagram.

    Post Cleanup:
        After posting the image, you can optionally delete the image to save space.

Required Libraries:

You need to install the following Python packages to run the script:

pip install requests Pillow instabot

Handling Random Cartoon Images:

    Cartoon Images: You can create or download a set of cartoon images and place them in the cartoons folder. If you want to fetch random cartoon images via an API, you can use services like DeepAI's CartoonGAN to generate cartoon images programmatically, or use any public domain cartoon images.

    Custom Image Sources: For more sophisticated image creation, you can use machine learning tools such as DeepArt or Style Transfer techniques to transform images into cartoon-like visuals.

Final Steps:

    Set up your Instagram credentials in the script.
    Add Text Files and Cartoons: Place your text files in the texts/ folder and cartoon images in the cartoons/ folder.
    Run the Script: The script will automatically process all text files, generate images with text, and post them to Instagram.

Improvements:

    Error Handling: Implement error handling for Instagram API failures, file handling issues, etc.
    Text Formatting: Use advanced text formatting libraries to ensure the text fits well on various images.
    API for Cartoon Images: Integrate with an external service that generates cartoon images on demand, if you don't have a set of predefined images.
    ===========
To generate videos with a voiceover of various artists based on text content stored in a folder, we can break the task into several key steps:

    Reading the text content from text files.
    Converting text to speech using a voice synthesis library (like Google Text-to-Speech or pyttsx3).
    Generating video content using a combination of static images or slides and the generated voiceover.
    Combining voice and video into a final video file using MoviePy.

Steps Breakdown:

    Reading Text: We'll read each text file stored in a folder.
    Text-to-Speech: We'll use Google Text-to-Speech (gTTS) to generate the voiceover. Optionally, you can use different voices by setting different languages or voices in the gTTS library or by using services like Amazon Polly.
    Video Generation: We'll use MoviePy to create videos, add images or slides, and sync the audio to the video.
    Voiceover Artists: You can specify different voices or use various TTS engines to mimic different artists' voices.

Requirements:

    gTTS: Google Text-to-Speech to convert text into speech.
    MoviePy: To create and edit videos.
    os and shutil: For file management.
    pygame: To handle some of the visual elements (e.g., adding text to video).
    Voice selection: You can adjust the TTS voice to simulate different "artists" if using services like Amazon Polly or gTTS's language features.

Installation of Required Libraries:

pip install gtts moviepy pygame

Python Code:

import os
from gtts import gTTS
from moviepy.editor import *
import pygame
from datetime import datetime

# Function to load text from file
def load_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to convert text to speech
def text_to_speech(text, output_audio_path, language='en'):
    tts = gTTS(text=text, lang=language)
    tts.save(output_audio_path)

# Function to create a video with an image and a voiceover
def create_video_with_voiceover(text, image_path="default_image.jpg", audio_path="voiceover.mp3", output_video_path="output_video.mp4"):
    # Load the image
    image_clip = ImageClip(image_path)
    image_clip = image_clip.set_duration(10)  # Duration of the image being displayed

    # Load the audio (voiceover)
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to match the image duration
    audio_clip = audio_clip.subclip(0, image_clip.duration)

    # Set the audio to the image video
    video_clip = image_clip.set_audio(audio_clip)

    # Write the final video to a file
    video_clip.write_videofile(output_video_path, codec="libx264", fps=24)

# Function to generate videos and post them
def generate_video_from_text_files():
    # Define the folder containing text files
    text_folder = "texts"
    
    # List all text files in the folder
    text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]
    
    for text_file in text_files:
        # Load the content of the text file
        text_path = os.path.join(text_folder, text_file)
        text_content = load_text_from_file(text_path)
        
        # Convert text to speech (voiceover)
        audio_file = f"audio/{text_file.split('.')[0]}.mp3"
        text_to_speech(text_content, audio_file, language='en')
        
        # Select a random image for the video
        random_image = "images/random_image.jpg"  # Default image (you can randomize this)

        # Generate the video with the text content as a voiceover
        video_file = f"videos/video_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        create_video_with_voiceover(text_content, image_path=random_image, audio_path=audio_file, output_video_path=video_file)
        
        # Optional: Remove audio file after video is created
        os.remove(audio_file)

        print(f"Video created: {video_file}")

# Function to create folders if they don't exist
def create_folders():
    os.makedirs("texts", exist_ok=True)  # Folder for text files
    os.makedirs("images", exist_ok=True)  # Folder for image files
    os.makedirs("audio", exist_ok=True)  # Folder for audio files (voiceovers)
    os.makedirs("videos", exist_ok=True)  # Folder for output videos

if __name__ == "__main__":
    create_folders()
    generate_video_from_text_files()

Explanation of Code:

    Load Text: The load_text_from_file function reads the content of each .txt file stored in the texts folder.

    Text-to-Speech:
        The text_to_speech function converts the text content into speech using Google Text-to-Speech (gTTS).
        You can change the language and accent of the voice by adjusting the lang parameter in the gTTS library (e.g., using lang='en' for English, or lang='en-us' for American English).
        The audio file is saved in the audio folder.

    Create Video:
        The create_video_with_voiceover function takes an image and a voiceover audio, combines them, and generates a video.
        The video is created with the image being displayed for the duration of the voiceover.
        MoviePy is used to create the video, synchronize the audio, and export the final video.

    Folder Management:
        The create_folders function ensures that necessary folders for storing text files, images, audio files, and generated videos are created if they don’t already exist.

    Processing Files:
        The generate_video_from_text_files function iterates through all .txt files in the texts folder, generates a voiceover for each, creates a video with a random or default image, and saves the output in the videos folder.

Customizations and Enhancements:

    Custom Images: If you have specific images for each video, you can add a random image selection mechanism or store corresponding images for each text file.
    Voice Variations: If you want different "artists" to voiceover the text, you can either select different TTS languages or voices available in the TTS system you are using.
    Image Duration: Adjust the duration of the image on-screen to match the length of the voiceover by using set_duration in MoviePy.
    Text Overlay on Image: You could add text overlay to the video as well (in addition to the voiceover) using MoviePy’s text functionalities.

Additional Libraries (Optional):

    pyttsx3: If you want to generate speech using a local offline engine rather than relying on Google TTS.
    Amazon Polly: For a more professional voiceover with a variety of voices.

Running the Code:

    Place your .txt files in the texts folder.
    Ensure you have images in the images folder or let the code choose randomly from the folder.
    Run the script, and it will generate videos with the corresponding voiceover from text files and save them to the videos folder.

This script provides a simple framework for generating videos with voiceovers using text files and can be expanded with additional features, such as adding more advanced images, multiple voices, or even syncing the voice to text in the video.
