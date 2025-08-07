import os
import cv2
import json
import re
import base64
from uuid import uuid4
from tqdm import tqdm
import time
from litellm import completion

# Configuration
VIDEO_PATH = "input_your_video_path_here"  # Replace with your video path
OUTPUT_FILE_PATH = os.path.join(VIDEO_PATH, "litellm_responses.json")
MCQ_JSON_FILE = "input_mcq_json_file_path_here"  # Replace with your MCQ JSON file path

# Set API keys
# os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# Model selection
MODEL_NAME = "gemini/gemini-1.5-flash"  # or "gpt-4-vision-preview" or "claude-3-opus-20240229"

def extract_video_frames(video_path, sample_frequency=50, max_frame_num=10):
    video = cv2.VideoCapture(video_path)
    frames_base64 = []
    frame_count = 0

    if not video.isOpened():
        print(f"Skipping video {video_path}: could not open file.")
        return None

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        if frame_count % sample_frequency == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(frame_base64)
            print(f"Processed frame {frame_count}")

        frame_count += 1

    video.release()

    if len(frames_base64) > max_frame_num:
        step = len(frames_base64) // max_frame_num
        frames_base64 = [frames_base64[i] for i in range(0, len(frames_base64), step)][:max_frame_num]

    print(f"{len(frames_base64)} frames extracted.")
    return frames_base64

def create_vision_messages(prompt, frames_base64):
    content = []
    
    for i, frame_base64 in enumerate(frames_base64):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_base64}"
            }
        })
    
    content.append({
        "type": "text",
        "text": prompt
    })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    return messages

def chat_with_model(prompt, frames_base64):
    try:
        messages = create_vision_messages(prompt, frames_base64)
        
        response = completion(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        
        time.sleep(3)  # Rate limiting
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error communicating with model: {e}")
        return None

def process_video_questions(video_data, video_path):
    results = []
    
    frames_base64 = extract_video_frames(video_path)
    
    if not frames_base64:
        return results
    
    for question_data in tqdm(video_data["data"], desc="Processing questions", unit="question", leave=False):
        question_id = str(uuid4())
        question = question_data["question"]
        original_answer = question_data["answer"]
        
        prompt = (
            f"{question} You must respond with exclusively the letter that corresponds to the correct answer: A, B, C, or D."
            " Return only a single letter (A, B, C, or D) with no additional text, symbols, or comments."
            " Must not respond with null, nothing, or any other characters—only return one of the letters."
            " You must strictly follow the above instructions."
        )
        
        response = chat_with_model(prompt, frames_base64)
        
        selected_option = None
        if response:
            match = re.search(r"\b([A-D])\b", response)
            if match:
                selected_option = match.group(1)
        
        output_record = {
            "id": video_path,
            "question_id": question_id,
            "question": question,
            "response": selected_option,
            "answer": original_answer,
            "model": MODEL_NAME
        }
        
        results.append(output_record)
    
    return results

def save_results(results):
    try:
        if os.path.exists(OUTPUT_FILE_PATH):
            with open(OUTPUT_FILE_PATH, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        existing_data.extend(results)
        
        with open(OUTPUT_FILE_PATH, "w") as f:
            json.dump(existing_data, f, indent=4)
            
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump([], f)
    
    try:
        with open(MCQ_JSON_FILE, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error reading MCQ file: {e}")
        return
    
    for item in tqdm(json_data, desc="Processing videos", unit="video"):
        video_path = os.path.join(VIDEO_PATH, item["id"])
        
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        
        results = process_video_questions(item, video_path)
        
        if results:
            save_results(results)
            print(f"Saved {len(results)} results")

if __name__ == "__main__":
    main()