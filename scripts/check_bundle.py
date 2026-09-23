"""Verify released artifacts and reproduce the paper's three main-result rows."""

import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEM_ID_NAME = "sentence-t5-base_rqkmeans_3x256_psid.sem_ids"
DOMAINS = {
    "industrial": {
        "category": "Industrial_and_Scientific",
        "label": "Industrial and Scientific",
        "samples": 50985,
        "alpha": 3.0,
        "data_hashes": {
            "id_mapping.json": "73a63ebef9be8d1d5e824c57e063c4bae5cce13b03f2e463f4f546de5dd4d103",
            "all_item_seqs.json": "41c6063f996dbec6114340f1223915f0b660f2215be8380946b1125179480a2b",
            "metadata.sentence.json": "1a08a0b52a006bce476aa4de3ba553387c27039735294f638f898fe5c5077b6f",
            SEM_ID_NAME: "6816772c9a32289cdf3d1b4ba7dda41ece8b62fa736802f3646691e52076cbee",
        },
        "release_hashes": {
            "checkpoint.pth": "66da41d01acd61831f3e222087f5662a24d52309d97878a9d2d66cfe002b6bc9",
            "processed_sid.sem_ids": "6816772c9a32289cdf3d1b4ba7dda41ece8b62fa736802f3646691e52076cbee",
            "sid/level_token_ids.pt": "ccc3b0b80df16e7dd9087ffed7b367013cbcaf026b7633d5f2c5e4d34abc2f12",
            "sid/valid_prefix_trie.json": "d36c1a377e3f4b881156f44822ac0132f4c49df9795d62976191cbe0216ace4f",
        },
        "generator": {"ndcg@5": 0.0216304436, "ndcg@10": 0.0271996558,
                      "recall@5": 0.0329508670, "recall@10": 0.0502500720},
        "full": {"ndcg@5": 0.0257368901, "ndcg@10": 0.0309455701,
                 "recall@5": 0.0374423850, "recall@10": 0.0536432284},
    },
    "musical": {
        "category": "Musical_Instruments",
        "label": "Musical Instruments",
        "samples": 57439,
        "alpha": 1.5,
        "data_hashes": {
            "id_mapping.json": "6525e3f586ec8741ac1c26f2f9abc4fff703ff4809407b855c481d84b7b7c3da",
            "all_item_seqs.json": "d9c5020f7370c4fd77a26f69ff19685a8256adab404d12f1f163b5dcbab9be36",
            "metadata.sentence.json": "c0539d77aa1673570a3ae083e75845454f3bc6ca0e5609da27d8c5896ed75ab7",
            SEM_ID_NAME: "1ec0363edd588f2a3b2a28a9cf6ce2f31eeb5c06422c10b441fcacd0ce201b9b",
        },
        "release_hashes": {
            "checkpoint.pth": "765338787fbbcb5222cd3dfda400fd00c277f28cedc1c0c785840182ade85e70",
            "processed_sid.sem_ids": "1ec0363edd588f2a3b2a28a9cf6ce2f31eeb5c06422c10b441fcacd0ce201b9b",
            "sid/level_token_ids.pt": "f5e201513030dbaeebbfe7727bcf4cf223e2f08a2cffd56fc55251c55309fe9e",
            "sid/valid_prefix_trie.json": "edc40a9cfa48526599768270eb7058c2eed562e3056323325804f9f9d25a5d5d",
        },
        "generator": {"ndcg@5": 0.0280908346, "ndcg@10": 0.0352846831,
                      "recall@5": 0.0424624383, "recall@10": 0.0648165867},
        "full": {"ndcg@5": 0.0323835596, "ndcg@10": 0.0395150915,
                 "recall@5": 0.0472153067, "recall@10": 0.0694475879},
    },
    "video": {
        "category": "Video_Games",
        "label": "Video Games",
        "samples": 94762,
        "alpha": 3.0,
        "data_hashes": {
            "id_mapping.json": "8314498c232313fa8b53c57da670f2f5a020f29cfdb038e8427d63ca06f9b1b8",
            "all_item_seqs.json": "8fb6640673f3548d25ec72b123bfb43dcf0eae09dfa0de8a52db8cc8916fc158",
            "metadata.sentence.json": "0a6711cfecd36c3cf78a642f1b05a66535d0a014a1063ff86e3f2b9491a9d2ae",
            SEM_ID_NAME: "e475ae082141b4ad52fb298d9fe3075ece38564e78230f8c3e530ccae5b3eccf",
        },
        "release_hashes": {
            "checkpoint.pth": "dc74c5febc39440c1206a73c9efae7b166e03a4a17d08b4ec8c9b9aebea0bf1c",
            "processed_sid.sem_ids": "e475ae082141b4ad52fb298d9fe3075ece38564e78230f8c3e530ccae5b3eccf",
            "sid/level_token_ids.pt": "e9b9c07da7ae714107111a1592ef5e14c80db9bb67b719edc7385d5488d57a68",
            "sid/valid_prefix_trie.json": "a3e3d0edc533914d340a09c684b26c8a0caa32564492edd9584849b81bcddbed",
        },
        "generator": {"ndcg@5": 0.0414204523, "ndcg@10": 0.0527197272,
                      "recall@5": 0.0628733039, "recall@10": 0.0980456322},
        "full": {"ndcg@5": 0.0510276696, "ndcg@10": 0.0617531642,
                 "recall@5": 0.0740486693, "recall@10": 0.1073426057},
    },
}


def check_data(slug):
    spec = DOMAINS[slug]
    cache = ROOT / "cache/AmazonReviews2023" / spec["category"] / "processed"
    bundle = ROOT / "release" / slug
    for directory, expected_files in (
        (cache, spec["data_hashes"]), (bundle, spec["release_hashes"])
    ):
        for name, expected in expected_files.items():
            path = directory / name
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise SystemExit(f"Artifact mismatch: {path}")
        print(f"verified {slug}: {len(expected_files)} files under {directory}")


def read_result(path):
    raw = Path(path).read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    for line in reversed(raw.decode(encoding).splitlines()):
        if line.startswith("COLAGR_RESULT="):
            return json.loads(line.split("=", 1)[1])
    raise SystemExit(f"Missing CoLeaf result: {path}")


def check_results(slug, validation_path, test_path):
    spec = DOMAINS[slug]
    validation = read_result(validation_path)
    test = read_result(test_path)
    alpha = validation["top_configs"][0]["alpha"]
    if alpha != spec["alpha"] or test["top_configs"][0]["alpha"] != alpha:
        raise SystemExit(f"Unexpected {slug} validation-selected alpha: {alpha}")
    if test["samples"] != spec["samples"]:
        raise SystemExit(f"Unexpected {slug} test sample count: {test['samples']}")
    print(f"{spec['label']} | test users={test['samples']} | alpha={alpha}")
    print("Model      Metric       Paper       Reproduced")
    for name, actual_metrics, expected_metrics in (
        ("Generator", test["base"], spec["generator"]),
        ("CoLaGR", test["top_configs"][0], spec["full"]),
    ):
        for metric, expected in expected_metrics.items():
            actual = actual_metrics[metric]
            print(f"{name:<10} {metric:<11} {expected:.6f}    {actual:.6f}")
            if abs(actual - expected) > 5e-5:
                raise SystemExit(f"Metric mismatch: {slug} {name} {metric}")
    return test["top_configs"][0]


def write_table(slugs):
    rows = []
    for slug in ("musical", "industrial", "video"):
        if slug not in slugs:
            continue
        out = ROOT / "results" / slug / "coleaf"
        metrics = check_results(slug, out / "validation.log", out / "test.log")
        rows.append((DOMAINS[slug]["label"], *(
            metrics[key] for key in ("recall@5", "recall@10", "ndcg@5", "ndcg@10")
        )))
    path = ROOT / "results/main_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Dataset", "R@5", "R@10", "N@5", "N@10"))
        writer.writerows((row[0], *(f"{value:.6f}" for value in row[1:])) for row in rows)
    print(f"Main-table rows saved to {path}")
    for row in rows:
        print(f"{row[0]:<26} " + "  ".join(f"{value:.4f}" for value in row[1:]))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "data":
        check_data(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == "results":
        check_results(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) >= 3 and sys.argv[1] == "table":
        write_table(set(sys.argv[2:]))
    else:
        raise SystemExit("Usage: check_bundle.py data SLUG | results SLUG VAL TEST | table SLUG...")


if __name__ == "__main__":
    main()
