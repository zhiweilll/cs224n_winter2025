#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2023-2024: Homework 2
evaluate.py: Evaluate a saved model on test data
Zhiwei Liang
"""
import argparse
import os
import torch

from parser_model import ParserModel
from utils.parser_utils import load_and_preprocess_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate saved model')
    parser.add_argument('-m', '--model', type=str, required=True, help='path to model weights file')
    parser.add_argument('-d', '--debug', action='store_true', help='whether to use debug mode')
    parser.add_argument('-o', '--output', type=str, default=None, help='path to save results (default: same dir as model)')
    args = parser.parse_args()

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser_obj, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(args.debug)

    print("Creating model...")
    model = ParserModel(embeddings)
    parser_obj.model = model

    print(80 * "=")
    print("LOADING MODEL")
    print(80 * "=")
    print(f"Loading model from: {args.model}")
    parser_obj.model.load_state_dict(torch.load(args.model))
    parser_obj.model.eval()

    print(80 * "=")
    print("EVALUATING ON TEST SET")
    print(80 * "=")
    UAS, dependencies = parser_obj.parse(test_data)
    print(f"- test UAS: {UAS * 100.0:.2f}%")

    # Save results
    if args.output:
        result_path = args.output
    else:
        # Auto-save to same directory as model
        model_dir = os.path.dirname(args.model)
        result_path = os.path.join(model_dir, "test_results.txt")
    
    with open(result_path, 'w') as f:
        f.write(f"Test UAS: {UAS * 100.0:.2f}%\n")
        f.write(f"Test UAS (raw): {UAS}\n")
    
    print(f"\nResults saved to: {result_path}")
    print("Done!")
