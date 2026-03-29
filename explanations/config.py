"""
Configuration for explanation generation.
Contains dataset metadata, language mappings, and task descriptions.
"""

import json
import os
from pathlib import Path

# Base paths
BASE_PATH = os.environ.get("MEMELENS_ROOT", ".")
DATA_PATH = os.path.join(BASE_PATH, "data/Unified_Labels_FullPath")
OUTPUT_PATH = os.path.join(BASE_PATH, "Explanation")
ENV_FILE = os.environ.get("MEMELENS_ENV_FILE", ".env")

# Load task definitions from JSON file
TASK_DEFINITION_FILE = os.path.join(BASE_PATH, "Explanation/task_definition.json")
try:
    with open(TASK_DEFINITION_FILE, "r", encoding="utf-8") as f:
        TASK_DEFINITIONS = json.load(f)
except FileNotFoundError:
    print(f"Warning: Task definition file not found at {TASK_DEFINITION_FILE}")
    TASK_DEFINITIONS = {}

# Datasets to skip (empty - process all datasets)
SKIP_DATASETS = []

# Language codes for native language names
LANGUAGE_MAP = {
    "en": "English",
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "es": "Spanish",
    "hi": "Hindi",
    "zh": "Chinese",
    "ro": "Romanian",
    "ru": "Russian",
}

# Dataset metadata: language, task type, labels
DATASET_CONFIG = {
    # ============ English Classification Datasets ============
    "Harmful_Covid_en__HarMeme": {
        "language": "en",
        "task": "COVID-19 meme harm classification",
        "labels": ["not-harmful", "harmful"],
        "description": "Classifies whether a COVID-19 related meme is harmful (spreads misinformation, promotes dangerous behaviors) or not.",
    },
    "Harmful_en__HarMeme": {
        "language": "en",
        "task": "Meme harm classification",
        "labels": ["not-harmful", "partially-harmful", "very-harmful"],
        "description": "Classifies the level of harmfulness in memes into three categories: not harmful, partially harmful, or very harmful.",
    },
    "Hateful_en_FHM": {
        "language": "en",
        "task": "Hateful meme detection",
        "labels": ["not-hateful", "hateful"],
        "description": "Detects whether a meme contains hateful content.",
    },
    "Hateful_en__MMHS": {
        "language": "en",
        "task": "Hateful meme detection",
        "labels": ["not-hateful", "hateful"],
        "description": "Detects hateful content in English memes from the MMHS150K dataset.",
    },
    "Hateful_en__MIMIC_Islamophpbia": {
        "language": "en",
        "task": "Islamophobia detection",
        "labels": ["not-hateful", "Hateful"],
        "description": "Detects Islamophobic content in memes.",
    },
    "Target_Covid_en__HarMeme": {
        "language": "en",
        "task": "COVID-19 meme target identification",
        "labels": ["individual", "organization", "community", "society"],
        "description": "Identifies the target of harm in COVID-19 related memes.",
    },
    "Target_en__HarMeme": {
        "language": "en",
        "task": "Meme target identification",
        "labels": ["individual", "organization", "community", "society"],
        "description": "Identifies the target of harmful content in memes.",
    },
    "humour_en__memotion": {
        "language": "en",
        "task": "Humor detection",
        "labels": ["not-funny", "funny", "very-funny", "hilarious"],
        "description": "Classifies the level of humor in memes.",
    },
    "intention_detection_en__MET_Meme": {
        "language": "en",
        "task": "Intention detection",
        "labels": ["motivational", "satirical", "informative", "provocative"],
        "description": "Detects the intention behind the meme.",
    },
    "metaphor_occurrence_en__MET_Meme": {
        "language": "en",
        "task": "Metaphor detection",
        "labels": ["no-metaphor", "metaphor"],
        "description": "Detects whether a meme contains metaphorical content.",
    },
    "misogynous_en__MAMI": {
        "language": "en",
        "task": "Misogyny detection",
        "labels": ["not-misogynistic", "misogynistic"],
        "description": "Detects misogynistic content in memes.",
    },
    "motivational_en__memotion": {
        "language": "en",
        "task": "Motivational content detection",
        "labels": ["not-motivational", "motivational"],
        "description": "Detects whether a meme is motivational.",
    },
    "objectification_en__MAMI": {
        "language": "en",
        "task": "Objectification detection",
        "labels": ["not-objectifying", "objectifying"],
        "description": "Detects objectification of women in memes.",
    },
    "offensive_en__memotion": {
        "language": "en",
        "task": "Offensive content detection",
        "labels": [
            "not-offensive",
            "slightly-offensive",
            "very-offensive",
            "hateful-offensive",
        ],
        "description": "Classifies the level of offensiveness in memes.",
    },
    "offensiveness_detection_en__MET_Meme": {
        "language": "en",
        "task": "Offensiveness detection",
        "labels": ["not-offensive", "offensive"],
        "description": "Detects offensive content in memes.",
    },
    "overall_sentiment_en__memotion": {
        "language": "en",
        "task": "Overall sentiment classification",
        "labels": ["very-negative", "negative", "neutral", "positive", "very-positive"],
        "description": "Classifies the overall sentiment of memes.",
    },
    "sarcasm_en__memotion": {
        "language": "en",
        "task": "Sarcasm detection",
        "labels": [
            "not-sarcastic",
            "general-sarcasm",
            "twisted-sarcasm",
            "very-twisted-sarcasm",
        ],
        "description": "Detects and classifies sarcasm in memes.",
    },
    "sentiment_category_en__MET_Meme": {
        "language": "en",
        "task": "Sentiment category classification",
        "labels": ["negative", "neutral", "positive"],
        "description": "Classifies the sentiment category of memes.",
    },
    "sentiment_degree_en__MET_Meme": {
        "language": "en",
        "task": "Sentiment degree classification",
        "labels": ["slightly-negative", "negative", "slightly-positive", "positive"],
        "description": "Classifies the degree of sentiment in memes.",
    },
    "shaming_en__MAMI": {
        "language": "en",
        "task": "Shaming detection",
        "labels": ["not-shaming", "shaming"],
        "description": "Detects shaming content in memes.",
    },
    "stereotype_en__MAMI": {
        "language": "en",
        "task": "Stereotype detection",
        "labels": ["not-stereotyping", "stereotyping"],
        "description": "Detects stereotyping content in memes.",
    },
    "violence_en__MAMI": {
        "language": "en",
        "task": "Violence detection",
        "labels": ["not-violent", "violent"],
        "description": "Detects violent content in memes.",
    },
    # ============ Arabic Datasets ============
    "Hateful_ar__Prop2Hate-Meme": {
        "language": "ar",
        "task": "Arabic hateful meme detection",
        "labels": ["not-hateful", "Hateful"],
        "native_labels": ["غير كراهية", "كراهية"],
        "description": "Detects hateful content in Arabic memes.",
    },
    "propoganda_ar_ArMeme": {
        "language": "ar",
        "task": "Arabic propaganda detection",
        "labels": ["not-propaganda", "propaganda"],
        "native_labels": ["ليست دعاية", "دعاية"],
        "description": "Detects propaganda content in Arabic memes.",
    },
    # ============ Bengali Datasets ============
    "Hateful_bn__MUTE": {
        "language": "bn",
        "task": "Bengali hateful meme detection",
        "labels": ["not-hateful", "hateful"],
        "native_labels": ["বিদ্বেষমূলক নয়", "বিদ্বেষমূলক"],
        "description": "Detects hateful content in Bengali memes.",
    },
    "abuse_bn__BanglaAbuseMeme": {
        "language": "bn",
        "task": "Bengali abuse detection",
        "labels": ["not-abusive", "abusive"],
        "native_labels": ["অপব্যবহারমূলক নয়", "অপব্যবহারমূলক"],
        "description": "Detects abusive content in Bengali memes.",
    },
    "sarcasm_bn__BanglaAbuseMeme": {
        "language": "bn",
        "task": "Bengali sarcasm detection",
        "labels": ["not-sarcastic", "sarcastic"],
        "native_labels": ["বিদ্রূপাত্মক নয়", "বিদ্রূপাত্মক"],
        "description": "Detects sarcasm in Bengali memes.",
    },
    "sentiment_bn__BanglaAbuseMeme": {
        "language": "bn",
        "task": "Bengali sentiment classification",
        "labels": ["negative", "neutral", "positive"],
        "native_labels": ["নেতিবাচক", "নিরপেক্ষ", "ইতিবাচক"],
        "description": "Classifies sentiment in Bengali memes.",
    },
    "vulgar_bn__BanglaAbuseMeme": {
        "language": "bn",
        "task": "Bengali vulgar content detection",
        "labels": ["not-vulgar", "vulgar"],
        "native_labels": ["অশ্লীল নয়", "অশ্লীল"],
        "description": "Detects vulgar content in Bengali memes.",
    },
    # ============ German Datasets ============
    "Hateful_de__Multi3Hate": {
        "language": "de",
        "task": "German hateful meme detection",
        "labels": ["not-hateful", "hateful"],
        "native_labels": ["nicht hasserfüllt", "hasserfüllt"],
        "description": "Detects hateful content in German memes.",
    },
    # ============ Spanish Datasets ============
    "Hateful_es__Multi3Hate": {
        "language": "es",
        "task": "Spanish hateful meme detection",
        "labels": ["not-hateful", "hateful"],
        "native_labels": ["no odioso", "odioso"],
        "description": "Detects hateful content in Spanish memes.",
    },
    # ============ English Multi3Hate ============
    "Hateful_en__Multi3Hate": {
        "language": "en",
        "task": "Hateful meme detection",
        "labels": ["not-hateful", "hateful"],
        "description": "Detects hateful content in English memes.",
    },
    # ============ Hindi Datasets ============
    "Hateful_hi__Multi3Hate": {
        "language": "hi",
        "task": "Hindi hateful meme detection",
        "labels": ["not-hateful", "hateful"],
        "native_labels": ["घृणास्पद नहीं", "घृणास्पद"],
        "description": "Detects hateful content in Hindi memes.",
    },
    "Misogyny_Categories_hi_en__MIMIC2024": {
        "language": "hi",
        "task": "Hindi-English misogyny category classification",
        "labels": [
            "not-misogynistic",
            "shaming",
            "stereotyping",
            "objectification",
            "violence",
        ],
        "native_labels": [
            "स्त्री द्वेषी नहीं",
            "शर्मसार करना",
            "रूढ़िवादिता",
            "वस्तुकरण",
            "हिंसा",
        ],
        "description": "Classifies types of misogyny in Hindi-English memes.",
    },
    "Misogyny_hi_en__MIMIC2024": {
        "language": "hi",
        "task": "Hindi-English misogyny detection",
        "labels": ["not-misogynistic", "misogynistic"],
        "native_labels": ["स्त्री द्वेषी नहीं", "स्त्री द्वेषी"],
        "description": "Detects misogynistic content in Hindi-English memes.",
    },
    # ============ Chinese Datasets ============
    "Hateful_zh__Multi3Hate": {
        "language": "zh",
        "task": "Chinese hateful meme detection",
        "labels": ["not-hateful", "hateful"],
        "native_labels": ["无仇恨", "仇恨"],
        "description": "Detects hateful content in Chinese memes.",
    },
    "intention_detection_zh__MET_Meme": {
        "language": "zh",
        "task": "Chinese intention detection",
        "labels": ["motivational", "satirical", "informative", "provocative"],
        "native_labels": ["激励性", "讽刺性", "信息性", "挑衅性"],
        "description": "Detects the intention behind Chinese memes.",
    },
    "metaphor_occurrence_zh__MET_Meme": {
        "language": "zh",
        "task": "Chinese metaphor detection",
        "labels": ["no-metaphor", "metaphor"],
        "native_labels": ["无隐喻", "隐喻"],
        "description": "Detects metaphorical content in Chinese memes.",
    },
    "offensiveness_detection_zh__MET_Meme": {
        "language": "zh",
        "task": "Chinese offensiveness detection",
        "labels": ["not-offensive", "offensive"],
        "native_labels": ["无冒犯", "冒犯"],
        "description": "Detects offensive content in Chinese memes.",
    },
    "sentiment_category_zh__MET_Meme": {
        "language": "zh",
        "task": "Chinese sentiment category classification",
        "labels": ["negative", "neutral", "positive"],
        "native_labels": ["负面", "中性", "正面"],
        "description": "Classifies the sentiment category of Chinese memes.",
    },
    "sentiment_degree_zh__MET_Meme": {
        "language": "zh",
        "task": "Chinese sentiment degree classification",
        "labels": ["slightly-negative", "negative", "slightly-positive", "positive"],
        "native_labels": ["略微负面", "负面", "略微正面", "正面"],
        "description": "Classifies the degree of sentiment in Chinese memes.",
    },
    # ============ Romanian Datasets ============
    "deepfake_ro__RoMemes": {
        "language": "ro",
        "task": "Romanian deepfake detection",
        "labels": ["Real", "Fake", "DeepFake"],
        "native_labels": ["Real", "Fals", "DeepFake"],
        "description": "Detects deepfake content in Romanian memes.",
    },
    "emotion_ro__RoMemes": {
        "language": "ro",
        "task": "Romanian emotion classification",
        "labels": ["Anger", "Fear", "Joy", "Love", "Sadness", "Surprise"],
        "native_labels": [
            "Furie",
            "Frică",
            "Bucurie",
            "Dragoste",
            "Tristețe",
            "Surpriză",
        ],
        "description": "Classifies the emotional tone conveyed in Romanian memes across six basic emotions.",
    },
    "political_ro__RoMemes": {
        "language": "ro",
        "task": "Romanian political content detection",
        "labels": ["non-political", "political"],
        "native_labels": ["non-politic", "politic"],
        "description": "Detects political content in Romanian memes.",
    },
    "sentiment_ro__RoMemes": {
        "language": "ro",
        "task": "Romanian sentiment classification",
        "labels": ["negative", "neutral", "positive"],
        "native_labels": ["negativ", "neutru", "pozitiv"],
        "description": "Classifies sentiment in Romanian memes.",
    },
    # ============ Russian Datasets ============
    "toxic_ru__Toxic_Memes_Detection_Dataset": {
        "language": "ru",
        "task": "Russian toxic content detection",
        "labels": ["not-toxic", "toxic"],
        "native_labels": ["нетоксичный", "токсичный"],
        "description": "Detects toxic content in Russian memes.",
    },
}

# GPT-4 endpoint for batch API
GPT_DICT = {"CHAT_REQUEST_URL": "/v1/chat/completions"}
