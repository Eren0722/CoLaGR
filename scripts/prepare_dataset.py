"""Build an official benchmark cache without republishing its user records."""

import argparse

from colagr.common import build_config
from genrec.datasets.AmazonReviews2023.dataset import AmazonReviews2023


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("category", choices=(
        "Industrial_and_Scientific", "Musical_Instruments", "Video_Games"
    ))
    args = parser.parse_args()
    config = build_config(
        "CoLaGR", "AmazonReviews2023", {"category": args.category}
    )
    AmazonReviews2023(config)


if __name__ == "__main__":
    main()
