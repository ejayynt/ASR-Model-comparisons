import os
import json
import time
import csv
import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()
AUDIO_DIR = "data/audio/"
GROUND_TRUTH_FILE = "data/ground_truth.json"
RESULTS_FILE = "results.csv"

# Ensure you set these in your environment or a .env file
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "your_deepgram_key")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "your_sarvam_key")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_key")

# METRICS UTILITY
def calculate_err(transcript: str, target_entity: str) -> bool:
    """
    Entity Recovery Rate (ERR).
    Uses fuzzy matching to account for phonetic misspellings of Indian localities.
    """
    if not transcript or not target_entity:
        return False
    
    transcript_clean = transcript.lower()
    target_clean = target_entity.lower()
    
    # Direct match
    if target_clean in transcript_clean:
        return True
    
    # Fuzzy match for phonetic errors (e.g., "Koramangala" vs "Kormangala")
    words = transcript_clean.split()
    for word in words:
        similarity = SequenceMatcher(None, target_clean, word).ratio()
        if similarity > 0.85:
            return True
            
    return False

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculates the Word Error Rate (WER) natively using Levenshtein distance.
    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Initialize a dynamic programming matrix
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
        
    # Compute edit distance at the word level
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
            )
            
    total_errors = d[len(ref_words)][len(hyp_words)]
    total_ref_words = len(ref_words)
    
    # Handle edge case where reference is empty
    if total_ref_words == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
        
    wer = total_errors / total_ref_words
    return round(wer, 4)

# API WRAPPERS
def evaluate_deepgram(audio_path: str) -> dict:
    start_time = time.time()
    url = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }
    
    try:
        with open(audio_path, "rb") as audio:
            response = requests.post(url, headers=headers, data=audio)
            response.raise_for_status()
            data = response.json()
            transcript = data['results']['channels'][0]['alternatives'][0]['transcript']
    except Exception as e:
        transcript = f"ERROR: {str(e)}"
        
    latency = time.time() - start_time
    return {"transcript": transcript, "latency": latency}

def evaluate_sarvam(audio_path: str) -> dict:
    start_time = time.time()
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    
    try:
        with open(audio_path, "rb") as audio:
            files = {"file": (os.path.basename(audio_path), audio, "audio/wav")}
            data = {"language_code": "hi-IN"} # Can be dynamically set based on ground truth
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            transcript = response.json().get("transcript", "")
    except Exception as e:
        transcript = f"ERROR: {str(e)}"
        
    latency = time.time() - start_time
    return {"transcript": transcript, "latency": latency}

def evaluate_groq_whisper(audio_path: str) -> dict:
    start_time = time.time()
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    
    try:
        with open(audio_path, "rb") as audio:
            files = {"file": (os.path.basename(audio_path), audio, "audio/wav")}
            data = {"model": "whisper-large-v3"}
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            transcript = response.json().get("text", "")
    except Exception as e:
        transcript = f"ERROR: {str(e)}"
        
    latency = time.time() - start_time
    return {"transcript": transcript, "latency": latency}

# MAIN EXECUTION LOOP
def main():
    print("Initializing ASR Evaluation Pipeline...")
    
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Error: {GROUND_TRUTH_FILE} not found. Please generate the data first.")
        return

    with open(GROUND_TRUTH_FILE, "r") as f:
        test_cases = json.load(f)
        
    results_data = []
    total_cases = len(test_cases)
    
    print(f"Loaded {total_cases} test cases. Commencing API benchmarking...\n")
    
    for idx, case in enumerate(test_cases, 1):
        file_id = case["file_id"]
        target = case["target_entity"]
        condition = case["condition"]
        audio_path = os.path.join(AUDIO_DIR, file_id)
        
        print(f"[{idx}/{total_cases}] Processing: {file_id} | Target: {target}")
        
        if not os.path.exists(audio_path):
            print(f"  -> WARNING: Audio file missing: {audio_path}")
            continue
            
        # Execute models
        dg_res = evaluate_deepgram(audio_path)
        sv_res = evaluate_sarvam(audio_path)
        gq_res = evaluate_groq_whisper(audio_path)

        # Get ground truth
        true_text = case["true_transcript"]
        
        # Calculate ERR
        dg_err = calculate_err(dg_res["transcript"], target)
        sv_err = calculate_err(sv_res["transcript"], target)
        gq_err = calculate_err(gq_res["transcript"], target)
        
        # Calculate WER
        dg_wer = calculate_wer(true_text, dg_res["transcript"])
        sv_wer = calculate_wer(true_text, sv_res["transcript"])
        gq_wer = calculate_wer(true_text, gq_res["transcript"])
        
        # Compile row
        results_data.append({
            "file_id": file_id,
            "target_entity": target,
            "condition": condition,
            "deepgram_transcript": dg_res["transcript"],
            "deepgram_latency_s": round(dg_res["latency"], 3),
            "deepgram_err": dg_err,
            "deepgram_wer": dg_wer,
            "sarvam_transcript": sv_res["transcript"],
            "sarvam_latency_s": round(sv_res["latency"], 3),
            "sarvam_err": sv_err,
            "sarvam_wer": sv_wer,
            "groq_whisper_transcript": gq_res["transcript"],
            "groq_whisper_latency_s": round(gq_res["latency"], 3),
            "groq_whisper_err": gq_err,
            "groq_whisper_wer": gq_wer
        })

    # Export to CSV
    keys = results_data[0].keys()
    with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_data)
        
    print(f"\nEvaluation complete. Results exported to {RESULTS_FILE}")

if __name__ == "__main__":
    main()