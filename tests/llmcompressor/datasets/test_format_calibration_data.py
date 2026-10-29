from unittest.mock import patch

import pytest
from datasets import Dataset

import llmcompressor.datasets.utils as dataset_utils
from llmcompressor.args import DatasetArguments


def _rows_read_by_rank(args, dataset, rank, world_size):
    with (
        patch.object(dataset_utils, "_is_dist_and_same_ds", return_value=True),
        patch.object(dataset_utils.dist, "get_rank", return_value=rank),
        patch.object(dataset_utils.dist, "get_world_size", return_value=world_size),
    ):
        dataloader = dataset_utils.format_calibration_data(args, dataset, None)
    return [row for batch in dataloader for row in batch["row"].tolist()]


@pytest.mark.unit
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("world_size", [2, 3])
def test_partitioned_dataloader_reads_each_rank_partition(shuffle, world_size):
    dataset = Dataset.from_dict(
        {"input_ids": [[0] * (i + 1) for i in range(12)], "row": list(range(12))}
    )
    args = DatasetArguments(
        num_calibration_samples=12, shuffle_calibration_samples=shuffle
    )

    rows_by_rank = [
        _rows_read_by_rank(args, dataset, rank, world_size)
        for rank in range(world_size)
    ]

    # every row is read by exactly one rank
    all_rows = [row for rows in rows_by_rank for row in rows]
    assert sorted(all_rows) == list(range(12))
