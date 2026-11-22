import json

from irsig.cli import main


def test_demo_end_to_end(tmp_path):
    main(["demo", "--out", str(tmp_path), "--n-perm", "2"])
    m = json.loads((tmp_path / "train" / "metrics.json").read_text())
    assert m["cv_scheme"] == "leave-one-dataset-out"
    assert 0.0 <= m["primary_auroc"] <= 1.0
    for f in ["train/model.joblib", "train/roc.png", "apply/scores.csv", "apply/association_tests.csv",
              "pseudobulk/detected_genes.txt"]:
        assert (tmp_path / f).exists(), f
    truth = json.loads((tmp_path / "apply" / "simulation_ground_truth_check.json").read_text())
    assert truth["auroc_score_vs_hidden_simulated_ir_ALL_PBMC"] > 0.7
