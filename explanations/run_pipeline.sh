#!/bin/bash
#
# Meme Explanation Generation - Complete Pipeline
# ================================================
# This script handles the entire explanation generation workflow:
# 1. Generate batch files for all datasets
# 2. Submit to Azure OpenAI Batch API
# 3. Monitor status and download results
# 4. Merge explanations with original datasets
#
# Usage:
#   ./run_pipeline.sh                    # Run full pipeline
#   ./run_pipeline.sh generate           # Generate batch files only
#   ./run_pipeline.sh submit             # Submit batches
#   ./run_pipeline.sh status             # Check status
#   ./run_pipeline.sh download           # Download results
#   ./run_pipeline.sh merge              # Merge with datasets
#   ./run_pipeline.sh clean              # Clean all generated files
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
ENV_FILE="${MEMELENS_ENV_FILE:-.env}"
TRACKING_FILE="${SCRIPT_DIR}/logs/batch_tracking.txt"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_header() {
    echo "========================================================================"
    echo "  Meme Explanation Generation Pipeline"
    echo "========================================================================"
    echo ""
}

check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file not found: $ENV_FILE"
        exit 1
    fi
    log_info "Using env file: $ENV_FILE"
}

generate_batches() {
    log_step "Generating batch files for all datasets..."
    cd "$SRC_DIR"
    python 1_submit_batches.py --generate_only --env_file "$ENV_FILE"
    log_info "✓ Batch files generated"
}

submit_batches() {
    log_step "Submitting batches to Azure OpenAI..."
    cd "$SRC_DIR"
    python 1_submit_batches.py --env_file "$ENV_FILE"
    log_info "✓ Batches submitted"
}

check_status() {
    log_step "Checking batch status..."
    cd "$SRC_DIR"
    python 2_retrieve_results.py --status_only --env_file "$ENV_FILE"
}

download_results() {
    log_step "Downloading completed results..."
    cd "$SRC_DIR"
    python 2_retrieve_results.py --env_file "$ENV_FILE"
    log_info "✓ Results downloaded"
}

merge_results() {
    log_step "Merging explanations with original datasets..."
    cd "$SRC_DIR"
    python 3_merge_results.py
    log_info "✓ Merge complete"
}

wait_for_completion() {
    log_step "Monitoring batch completion (press Ctrl+C to stop)..."
    
    while true; do
        cd "$SRC_DIR"
        
        # Get status
        status_output=$(python 2_retrieve_results.py --status_only --env_file "$ENV_FILE" 2>&1)
        
        # Extract counts
        completed=$(echo "$status_output" | grep "Completed:" | awk '{print $2}' || echo "0")
        in_progress=$(echo "$status_output" | grep "In Progress:" | awk '{print $3}' || echo "0")
        failed=$(echo "$status_output" | grep "Failed:" | awk '{print $2}' || echo "0")
        
        log_info "Status: Completed=$completed, In Progress=$in_progress, Failed=$failed"
        
        if [ "$in_progress" = "0" ]; then
            log_info "All batches completed!"
            break
        fi
        
        log_info "Waiting 5 minutes before next check..."
        sleep 300  # 5 minutes
    done
}

clean_all() {
    log_warn "Cleaning all generated files..."
    rm -rf "${SCRIPT_DIR}/batch_files/"*
    rm -rf "${SCRIPT_DIR}/outputs/"*
    rm -rf "${SCRIPT_DIR}/merged_data/"*
    rm -rf "${SCRIPT_DIR}/logs/"*.json
    rm -rf "${SCRIPT_DIR}/logs/batch_tracking.txt"
    rm -rf "${SRC_DIR}/__pycache__"
    log_info "✓ Cleaned"
}

run_full_pipeline() {
    print_header
    check_env
    
    log_info "Starting full pipeline..."
    echo ""
    
    # Step 1: Generate and submit
    generate_batches
    echo ""
    submit_batches
    echo ""
    
    # Step 2: Wait for completion
    log_warn "Batches submitted. They will take several hours to complete."
    log_info "You can:"
    log_info "  1. Wait here and monitor (automatic check every 5 minutes)"
    log_info "  2. Exit (Ctrl+C) and come back later to run:"
    log_info "     ./run_pipeline.sh status    (to check)"
    log_info "     ./run_pipeline.sh download  (to download when ready)"
    log_info "     ./run_pipeline.sh merge     (to merge results)"
    echo ""
    
    read -p "Wait and monitor now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wait_for_completion
        echo ""
        download_results
        echo ""
        merge_results
        echo ""
        log_info "✓✓✓ Pipeline complete! Check merged_data/ for results."
    else
        log_info "Batches submitted. Run './run_pipeline.sh status' to check later."
    fi
}

# Main
case "${1:-}" in
    generate)
        print_header
        check_env
        generate_batches
        ;;
    submit)
        print_header
        check_env
        submit_batches
        ;;
    status)
        print_header
        check_env
        check_status
        ;;
    download)
        print_header
        check_env
        download_results
        ;;
    merge)
        print_header
        merge_results
        ;;
    clean)
        clean_all
        ;;
    wait)
        print_header
        check_env
        wait_for_completion
        ;;
    help|--help|-h)
        echo "Usage: ./run_pipeline.sh [command]"
        echo ""
        echo "Commands:"
        echo "  (none)      Run full pipeline (generate -> submit -> wait -> download -> merge)"
        echo "  generate    Generate batch files only"
        echo "  submit      Submit batches to Azure OpenAI"
        echo "  status      Check status of submitted batches"
        echo "  wait        Wait and monitor until completion"
        echo "  download    Download completed results"
        echo "  merge       Merge explanations with original datasets"
        echo "  clean       Clean all generated files"
        echo "  help        Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_pipeline.sh              # Run everything"
        echo "  ./run_pipeline.sh generate     # Just generate files"
        echo "  ./run_pipeline.sh status       # Check what's happening"
        ;;
    *)
        run_full_pipeline
        ;;
esac
