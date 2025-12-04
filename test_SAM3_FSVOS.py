import argparse
import random
import json
from SAM3_FSVOS import SAM3_FSVOS

def get_arguments():
        parser = argparse.ArgumentParser(description='FSVOS')
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--session_name", type=str, default=str(random.randbytes(4).hex()))
        parser.add_argument("--dataset_path", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default="./output")
        parser.add_argument("--group", type=int, default=1)
        parser.add_argument("--test_query_frame_num", type=int, default=None)
        parser.add_argument("--verbose", type=bool, default=False)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--crop_paste_support_to_query", type=bool, default=False)
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
    
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    sam3_predictor = SAM3_FSVOS(
        checkpoint=args.checkpoint,
        config=args.config,
        session_name=args.session_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        verbose=args.verbose,
        test_query_frame_num=args.test_query_frame_num,
    )

    mean_f, mean_j, score_dict = sam3_predictor.test(group=args.group, seed=args.seed, crop_paste_support_to_query=args.crop_paste_support_to_query)
    print(f"Group {args.group} - Mean F: {mean_f}, Mean J: {mean_j}")
    print(f"Detailed Scores: {json.dumps(score_dict, indent=4)}")


if __name__ == '__main__':
    main()