from stable_audio_tools.interface.gradio import create_ui
import json 

import torch

def main(args):
    torch.manual_seed(42)

    print(f"Loading model config from {args.model_config}")
    # Load config from json file
    with open(args.model_config) as f:
        model_config = json.load(f)


    interface = create_ui(model_config, args.ckpt_path, args.pretransform_ckpt_path)
    interface.queue()
    interface.launch(share=args.share, auth=(args.username, args.password) if args.username is not None else None)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run gradio interface')
    parser.add_argument('--model-config', type=str, help='Path to model config', required=True)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=True)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint', required=False)
    parser.add_argument('--username', type=str, help='Gradio username', required=False)
    parser.add_argument('--password', type=str, help='Gradio password', required=False)
    parser.add_argument('--share', action='store_true', help='Share the interface', required=False)
    args = parser.parse_args()
    main(args)