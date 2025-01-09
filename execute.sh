#!/bin/bash

# Navigate to inference directory
cd "benchmark and datasets"/benchmark/inference

# Set the model type: 'closed' for closed-source model, 'open' for open-source model
model_type="closed"  # Change to 'open' if using an open-source model

if [ "$model_type" = "closed" ]; then
    # For closed-source model
    echo "Please ensure you have filled in your API key and base_url in close_model.py before proceeding."
    
    # Set default parameters (modify these as needed)
    model_name="gpt-4o-mini"       # Required: specify your model name
    top_k=0                            # Optional: integer between 0-9; 0 means no RAG
    num_threads=10                      # Optional: integer between 1-32
    lib_name=""                        # Optional: specify the python library name
    answer_difficulty=""               # Optional: specify the difficulty level
    category=""                        # Optional: specify the question category
    question_type=""                   # Optional: specify the question type

    # Build the command
    cmd="python close_model.py --model_name \"$model_name\""
    if [ -n "$top_k" ]; then
        cmd="$cmd --top_k $top_k"
    fi
    if [ -n "$num_threads" ]; then
        cmd="$cmd --num_threads $num_threads"
    fi
    if [ -n "$lib_name" ]; then
        cmd="$cmd --lib_name \"$lib_name\""
    fi
    if [ -n "$answer_difficulty" ]; then
        cmd="$cmd --answer_difficulty \"$answer_difficulty\""
    fi
    if [ -n "$category" ]; then
        cmd="$cmd --category \"$category\""
    fi
    if [ -n "$question_type" ]; then
        cmd="$cmd --question_type \"$question_type\""
    fi

    # Run the command
    eval $cmd

elif [ "$model_type" = "open" ]; then
    # For open-source model
    echo "Please ensure you have modified the model path and model_settings path in open_model.py before proceeding."
    
    # Set default parameters (modify these as needed)
    model_name="your_model_name"       # Required: specify your model name
    model_setting="your_model_setting" # Required: specify your model setting
    top_k=0                            # Optional: integer between 0-9; 0 means no RAG
    lib_name=""                        # Optional: specify the python library name
    answer_difficulty=""               # Optional: specify the difficulty level
    category=""                        # Optional: specify the question category
    question_type=""                   # Optional: specify the question type

    # Build the command
    cmd="python open_model.py --model_name \"$model_name\" --model_setting \"$model_setting\""
    if [ -n "$top_k" ]; then
        cmd="$cmd --top_k $top_k"
    fi
    if [ -n "$lib_name" ]; then
        cmd="$cmd --lib_name \"$lib_name\""
    fi
    if [ -n "$answer_difficulty" ]; then
        cmd="$cmd --answer_difficulty \"$answer_difficulty\""
    fi
    if [ -n "$category" ]; then
        cmd="$cmd --category \"$category\""
    fi
    if [ -n "$question_type" ]; then
        cmd="$cmd --question_type \"$question_type\""
    fi

    # Run the command
    eval $cmd

else
    echo "Invalid model_type. Please set model_type to 'closed' or 'open'."
    exit 1
fi

# Navigate to evaluation/run directory
cd ../evaluation/run

# Run 1_run_test.py
python 1_run_test.py