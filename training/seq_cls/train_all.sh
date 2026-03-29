#!/bin/bash
#
# Master training script - runs all dataset training scripts sequentially
#
# Usage: 
#   ./train_all.sh              # Train all datasets
#   ./train_all.sh dataset1 dataset2  # Train specific datasets
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# List of all datasets
DATASETS=(
    "Harmful_Covid_en__HarMeme"
    "Harmful_en__HarMeme"
    "Hateful_bn__MUTE"
    "Hateful_de__Multi3Hate"
    "Hateful_en_FHM"
    "Hateful_en__MIMIC_Islamophpbia"
    "Hateful_en__Multi3Hate"
    "Hateful_es__Multi3Hate"
    "Hateful_hi__Multi3Hate"
    "Hateful_zh__Multi3Hate"
    "Misogyny_hi_en__MIMIC2024"
    "Target_Covid_en__HarMeme"
    "Target_en__HarMeme"
    "abuse_bn__BanglaAbuseMeme"
    "emotion_ro__RoMemes"
    "fakenews_ro__RoMemes"
    "humour_en__memotion"
    "intention_detection_en__MET_Meme"
    "intention_detection_zh__MET_Meme"
    "metaphor_occurrence_en__MET_Meme"
    "metaphor_occurrence_zh__MET_Meme"
    "misogynous_en__MAMI"
    "motivational_en__memotion"
    "objectification_en__MAMI"
    "offensive_en__memotion"
    "offensiveness_detection_en__MET_Meme"
    "offensiveness_detection_zh__MET_Meme"
    "overall_sentiment_en__memotion"
    "political_ro__RoMemes"
    "propoganda_ar_ArMeme"
    "sarcasm_bn__BanglaAbuseMeme"
    "sarcasm_en__memotion"
    "sentiment_bn__BanglaAbuseMeme"
    "sentiment_category_en__MET_Meme"
    "sentiment_category_zh__MET_Meme"
    "sentiment_degree_en__MET_Meme"
    "sentiment_degree_zh__MET_Meme"
    "sentiment_ro__RoMemes"
    "shaming_en__MAMI"
    "stereotype_en__MAMI"
    "toxic_ru__Toxic_Memes_Detection_Dataset"
    "violence_en__MAMI"
    "vulgar_bn__BanglaAbuseMeme"
)

# If arguments provided, use them as the dataset list
if [ $# -gt 0 ]; then
    DATASETS=("$@")
fi

echo "========================================"
echo "Training ${#DATASETS[@]} datasets"
echo "========================================"
echo ""

SUCCESS=0
FAILED=0

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training: $dataset"
    echo "========================================"
    
    SCRIPT="${SCRIPT_DIR}/train_${dataset}.sh"
    
    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: Script not found: $SCRIPT"
        ((FAILED++))
        continue
    fi
    
    # Make script executable
    chmod +x "$SCRIPT"
    
    # Run training script
    if bash "$SCRIPT"; then
        echo "✓ Successfully trained: $dataset"
        ((SUCCESS++))
    else
        echo "✗ Failed to train: $dataset"
        ((FAILED++))
    fi
done

echo ""
echo "========================================"
echo "Training Summary"
echo "========================================"
echo "✓ Successful: $SUCCESS"
echo "✗ Failed: $FAILED"
echo "Total: ${#DATASETS[@]}"
echo ""
