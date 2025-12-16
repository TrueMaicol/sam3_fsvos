import argparse
import random
import json
from SAM3_FSVOS import SAM3_FSVOS
from SAM3_FSVOS_TEXT import SAM3_FSVOS_TEXT

def get_arguments():
        parser = argparse.ArgumentParser(description='FSVOS')
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--session_name", type=str, default=str(random.randbytes(4).hex()))
        parser.add_argument("--dataset_path", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default="./output")
        parser.add_argument("--group", type=int, default=1)
        parser.add_argument("--test_query_frame_num", type=int, default=None)
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--crop_paste_support_to_query", action="store_true")
        # text_prompt -> prompt SAM3 with text
        parser.add_argument("--text_prompt", action="store_true")
        # use_support_visuals -> prompt SAM3 with support frames 
        parser.add_argument("--use_support_visuals", action="store_true")
        parser.add_argument("--nshot", type=int, default=5)
        parser.add_argument("--gen_labels", action="store_true")

        # VLM PARAMS
        parser.add_argument('--vlm_model_path', type=str, default="./models", help='Path to the VLM model directory')
        parser.add_argument('--prompt_type', type=str, default='contour', choices=['mask', 'bb', 'contour', 'ellipse'])
        parser.add_argument('--zoom_percentage', type=int, default=50)
        parser.add_argument('--color', type=str, default="red", choices=["red", "green", "blue"])
        parser.add_argument('--ensamble_prompts', action='store_true', help='Use multiple prompts for the same image')
        parser.add_argument('--ensamble_prompts_list', type=str, nargs="+", default=["bb", "contour", "ellipse"])
        parser.add_argument('--ensamble_zoom', action='store_true', help='Use multiple zoom percentages for the same image')
        parser.add_argument('--ensamble_zoom_list', type=int, nargs="+", default=[0, 30, 50])
        parser.add_argument('--ensamble_colors', action='store_true', help='Use multiple colors for the same image')
        parser.add_argument('--ensamble_colors_list', type=str, nargs="+", default=["red", "green", "blue"])
        parser.add_argument('--alpha_blending', type=float, default=0.5)
        parser.add_argument('--thickness', type=int, default=2)



        return parser.parse_args()


def youtube_fsvos_test(predictor):

    results = {}

    for j in range(1, 5):
        print(f"Started evaluation on fold {j}")

        mean_f, mean_j, score_dict = predictor.test(group=j)

        print(f"Fold {j}/4 results:")
        print(f"Group {j} - Mean F: {mean_f}, Mean J: {mean_j}")
        print(f"Detailed Scores: {json.dumps(score_dict, indent=4)} \n \n \n")

        results[f"fold_{j}"] = {
            "mean_f": mean_f,
            "mean_j": mean_j,
            "detailed_scores": score_dict
        }

    return results

def main():
    args = get_arguments()
    
    # Validate arguments
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    # if text_prompt is not set, use SAM3_FSVOS, if it's set use SAM3_FSVOS_TEXT
    if not args.text_prompt:
        print("Using SAM3_FSVOS")
        sam3_predictor = SAM3_FSVOS(
            checkpoint=args.checkpoint,
            session_name=args.session_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            verbose=args.verbose,
            test_query_frame_num=args.test_query_frame_num,
        )
        mean_f, mean_j, score_dict = sam3_predictor.test(group=args.group, seed=args.seed, crop_paste_support_to_query=args.crop_paste_support_to_query, nshot=args.nshot)
    else:
        print("Using SAM3_FSVOS_TEXT")

        assert not (args.gen_labels and args.nshot <= 0)

        sam3_predictor_text = SAM3_FSVOS_TEXT(
            checkpoint=args.checkpoint,
            session_name=args.session_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            verbose=args.verbose,
            test_query_frame_num=args.test_query_frame_num,
            args=args
        )
        mean_f, mean_j, score_dict = sam3_predictor_text.test(group=args.group, seed=args.seed, nshot=args.nshot)

    print(f"Group {args.group} - Mean F: {mean_f}, Mean J: {mean_j}")
    print(f"Detailed Scores: {json.dumps(score_dict, indent=4)}")


if __name__ == '__main__':
    main()