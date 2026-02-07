import pyarrow as pa
from src.bioml.data.splits import SplitConfig, add_split_column


def test_add_split_column():
    table = pa.Table.from_pylist(
        [{"person_id": 1, "x": 0}, {"person_id": 1, "x": 1}, {"person_id": 2, "x": 2}]
    )
    out = add_split_column(table, SplitConfig(train=0.5, val=0.25, test=0.25, seed=0))
    assert "split" in out.column_names
    assert set(out["split"].to_pylist()) <= {"train", "val", "test"}
